import logging
from pathlib import Path
import time
from argparse import Namespace
from tqdm import tqdm

import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.utils import ProjectConfiguration
from transformers import TrainingArguments
from datasets import load_dataset
from huggingface_hub import Repository
from torch.optim import AdamW
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, HfArgumentParser, get_scheduler, set_seed, pipeline

import peft
from peft import LoraConfig, get_peft_model


def create_unlearn_dataloader(tokenizer, dataset, fraction=1.0, batch_size=4):
    """
    Given the PKU dataset, create the dataloader on the unlearned harmful Q&A pairs.

    Args:
        tokenizer: Tokenizer.
        dataset: Loaded PKU dataset.
        fraction: <1 will do downsampling.
        batch_size: Batch size.

    Returns:
        Data loader of PKU harmful Q&A pairs.
    """

    # Preprocess function.
    def preprocess(examples):
        results = {"input_ids": [], "attention_mask": [], "start_locs": []}
        for i in range(len(examples["question"])):
            question = examples["question"][i]
            answer = examples["answer"][i]
            tokenized = tokenizer(f"{question}\n{answer}", truncation=True, padding="max_length")
            results["input_ids"].append(tokenized["input_ids"])
            results["attention_mask"].append(tokenized["attention_mask"])
            
            # test_text = f"{question}\n"
            # test_tokenized = tokenizer(
            #     test_text, truncation=True,
            #     #padding="max_length"  # NOTE: Comment out the argument provided in the original implementation because using this means len(test_tokenized["input_ids"]) is always the max_length.
            # )
            # results["start_locs"].append(len(test_tokenized["input_ids"]) - 1)

            # Need to set a different start index for left-padding?
            results["start_locs"].append(len(results["input_ids"]) - len(tokenizer(answer, truncation=True)))

        return results

    # Need to drop all original columns to emit more than one row for each original row https://huggingface.co/docs/datasets/about_map_batch#input-size-output-size.
    dataset = dataset.map(preprocess, batched=True, remove_columns=["filename", "question", "answer"])
    dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "start_locs"]
    )

    # Add labels and make it data loader.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=data_collator
    )

    return dataloader


def get_answer_loss(operation, batch, model, device="cuda:0"):
    """
    Compute the loss on the answer (i.e. y) part.

    Args:
        operation: either "ga" (gradient ascent) or "gd" (gradient descent).
        batch: A batch of data.
        model: The unlearned model.
        device: GPU device.

    Returns:
       The loss.
    """
    assert operation in ["ga", "gd"], "Operation must be either GA or GD."
    input_ids, attention_mask, start_locs, labels = (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["start_locs"],
        batch["labels"].to(device),
    )
    outputs = model(input_ids, attention_mask=attention_mask)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    # Shift one to predict next token.
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    losses = []
    for bid in range(input_ids.shape[0]):
        one_inp, one_st = input_ids[bid], start_locs[bid]

        # GA or GD.
        position_loss = loss_fct(shift_logits[bid], shift_labels[bid])
        if operation == "ga":  # Negative the direction for GA.
            position_loss = -position_loss

        # Simply put equal weights on all answers.
        position_weight = torch.zeros_like(one_inp)
        assert len(position_weight) == len(position_loss) + 1
        position_weight[one_st:] = 1  # only focus on answer part

        # Should not do this 0 filling for left-padding?
        # Ignore the padding part.
        # position_weight[one_inp == 1] = 0
        if position_weight.sum() > 0:
            position_weight = position_weight / position_weight.sum()

        one_loss = (position_weight[:-1] * position_loss).sum()
        losses.append(one_loss)
    final_loss = torch.stack(losses).mean()

    return final_loss


def get_grouped_params(model, args, no_decay=["bias", "ln_1.weight", "ln_2.weight", "ln_f.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": args.weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]


# args
parser = HfArgumentParser(TrainingArguments)
parser.add_argument("--data_dir", type=str, default="data/unlearning", help="Dataset dir.")
parser.add_argument("--model_name", type=str, default="codeparrot/codeparrot-small", help="Model name.")
parser.add_argument("--turn_off_lora", action="store_true", help="Turn off LoRA.")
parser.add_argument("--lora_r", type=int, default=8, help="r for LoRA. Ignored when turn_off_lora=True")
parser.add_argument("--tokenized", type=bool, default=False, help="Dataset is tokenized or not.")
parser.add_argument("--lambda", type=float, default=1e-1, help="Coefficient for maintain loss.")
parser.add_argument("--max_unlearn_loss", type=float, default=100, help="Maximum loss on bad samples to terminate.")
args = parser.parse_args()

name = "lr-{}_bs-{}_accsteps-{}_epochs-{}_maxsteps-{}_warmsteps-{}_lora-{}_seed-{}".format(
    args.learning_rate, args.per_device_train_batch_size, args.gradient_accumulation_steps, args.num_train_epochs,
    0 if args.max_steps is None or args.max_steps <= 0 else args.max_steps, args.warmup_steps,
    0 if args.turn_off_lora else args.lora_r, args.seed,
)

# directories
log_dir = Path(args.output_dir) / "logs"
model_dir = Path(args.output_dir) / "models"
log_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"{name}.log"
log_file.unlink(missing_ok=True)

# logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=log_file,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# accelerator
accelerator = Accelerator()
acc_state = {str(k): str(v) for k, v in accelerator.state.__dict__.items() if k not in args}
device = accelerator.device

args = Namespace(**vars(args), **acc_state)
logger.info(args)
assert args.distributed_type == "DistributedType.NO"

samples_per_step = accelerator.state.num_processes * args.per_device_train_batch_size
set_seed(args.seed)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(args.model_name)
if model.__class__.__name__.startswith("GPT2"):
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

if not args.turn_off_lora:
    config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=8,
        lora_dropout=0.01,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

print(tokenizer)
print(model)
train_dataset = load_dataset(path=args.data_dir, split="train")
train_dataset = train_dataset.shuffle()
print(train_dataset)
train_dataloader = create_unlearn_dataloader(tokenizer, dataset=train_dataset, batch_size=args.per_device_train_batch_size)
#print(train_dataloader)
optimizer = AdamW(get_grouped_params(model, args), lr=args.learning_rate)
lr_scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.warmup_steps,
    num_training_steps=args.num_train_epochs * (len(train_dataset) // (args.per_device_train_batch_size * args.gradient_accumulation_steps)) if args.max_steps is None or args.max_steps <= 0 else args.max_steps,
)
print(optimizer)
print(lr_scheduler)

accelerator.register_for_checkpointing(lr_scheduler)

def get_lr():
    return optimizer.param_groups[0]["lr"]

model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

model.train()
completed_steps = 0
accumulated_loss_unlearn = 0.
accumulated_loss_mainrain = 0.
accumulated_loss_total = 0.

logger.info(f"num_train_epochs {args.num_train_epochs}, max_steps: {args.max_steps}")
for epoch in range(1, int(args.num_train_epochs) + 1):
    for step, batch in enumerate(tqdm(train_dataloader), start=1):
        loss_total = get_answer_loss("ga", batch, model, device=device)
        accumulated_loss_unlearn += loss_total.item() / args.gradient_accumulation_steps

        # loss_maintain = <DO ANYTHING FUN>
        # accumulated_loss_maintian += loss_maintain.item() / args.gradient_accumulation_steps
        # loss_total += lambda * loss_maintain

        accumulated_loss_total += loss_total.item() / args.gradient_accumulation_steps
        accelerator.backward(loss_total)

        if step % args.gradient_accumulation_steps == 0:
            lr = get_lr()
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            logger.info(f"step: {step}, lr: {lr}, loss_total: {accumulated_loss_total}, loss_unlearn: {accumulated_loss_unlearn}, loss_maintain: {accumulated_loss_mainrain}")
            accumulated_loss_unlearn = 0.
            accumulated_loss_mainrain = 0.
            accumulated_loss_total = 0.
            completed_steps += 1

    if args.max_steps is not None and args.max_steps > 0 and args.max_stepscompleted_steps >= args.max_steps:
        logger.info(f"Max steps {args.max_steps} reached")
        break


accelerator.wait_for_everyone()
if not args.turn_off_lora:
    model = model.merge_and_unload()
unwrapped_model = accelerator.unwrap_model(model)
save_dir = model_dir / name
unwrapped_model.save_pretrained(save_dir, save_function=accelerator.save)
tokenizer.save_pretrained(save_dir, save_function=accelerator.save)

# Example of generation
prompt = "class DataUpdate(BaseDataUpdate):\n"
tokenizer = AutoTokenizer.from_pretrained(save_dir)
model = AutoModelForCausalLM.from_pretrained(save_dir)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
outputs = generator(prompt, max_length=256, num_return_sequences=10)
for i, output in enumerate(outputs):
    logger.info(f"Generated code {i + 1}")
    logger.info(output["generated_text"])