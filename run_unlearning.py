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
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, get_scheduler, set_seed

import peft
from peft import LoraConfig, get_peft_model


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
            tokenized (bool): If true we use a pretokenized dataset.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        tokenized=False,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.bos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.epoch = 0
        self.infinite = infinite
        self.current_size = 0
        self.tokenized = tokenized

        if self.tokenized:
            self.max_buffer_size = seq_length * num_of_sequences
            self.content_field = "input_ids"
        else:
            self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
            self.content_field = "content"

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        self.epoch += 1
                        logger.info(f"Dataset epoch: {self.epoch}")
                    else:
                        more_examples = False
                        break
            if self.tokenized:
                tokenized_inputs = buffer
            else:
                tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    self.current_size += 1
                    yield torch.tensor(input_ids)

    def shuffle(self, buffer_size=1000):
        return ShufflerIterDataPipe(self, buffer_size=buffer_size)


def create_dataloaders(args, tokenizer):
    train_data = load_dataset(path=args.data_dir, split="train")
    train_data = train_data.shuffle(seed=args.seed)
    # valid_data = load_dataset(args.dataset_name_valid, split="valid")
    train_dataset = ConstantLengthDataset(
        tokenizer, train_data, infinite=True, seq_length=args.seq_length, tokenized=args.tokenized
    )
    # valid_dataset = ConstantLengthDataset(
    #     tokenizer, valid_data, infinite=False, seq_length=args.seq_length, tokenized=args.tokenized
    # )
    train_dataset = train_dataset.shuffle(buffer_size=args.shuffle_buffer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True)
    eval_dataloader = None
    # eval_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size)
    return train_dataloader, eval_dataloader


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


def evaluate(args):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch, labels=batch)
        loss = outputs.loss.repeat(args.valid_batch_size)
        losses.append(accelerator.gather(loss))
        if args.max_eval_steps > 0 and step >= args.max_eval_steps:
            break
    losses = torch.cat(losses)
    loss = losses[: eval_dataloader.dataset.current_size].mean()
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()


# args
parser = HfArgumentParser(TrainingArguments)
parser.add_argument("--data_dir", type=str, default="data/unlearning", help="Dataset dir.")
parser.add_argument("--model_name", type=str, default="codeparrot/codeparrot-small", help="Model name.")
parser.add_argument("--turn_off_lora", type=bool, default=False, help="Turn off LoRA.")
parser.add_argument("--lora_r", type=int, default=8, help="r for LoRA. Ignored when turn_off_lora=True")
parser.add_argument("--tokenized", type=bool, default=False, help="Dataset is tokenized or not.")
parser.add_argument("--shuffle_buffer", type=int, default=10000, help="Size of buffer used to shuffle streaming dataset.")
parser.add_argument("--seq_length", type=int, default=1024, help="Sequence lengths used for training.")
parser.add_argument("--lambda", type=float, default=1e-1, help="Coefficient for maintain loss.")
args = parser.parse_args()

name = "lr-{}_bs-{}_accsteps-{}_maxsteps-{}_warmsteps-{}_lora-{}_seed-{}".format(
    args.learning_rate, args.per_device_train_batch_size, args.gradient_accumulation_steps, args.max_steps, args.warmup_steps,
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

args = Namespace(**vars(args), **acc_state)
logger.info(args)
assert args.distributed_type == "DistributedType.NO"

samples_per_step = accelerator.state.num_processes * args.per_device_train_batch_size
set_seed(args.seed)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(args.model_name)

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
train_dataloader, eval_dataloader = create_dataloaders(args, tokenizer)
#print(train_dataloader, eval_dataloader)
optimizer = AdamW(get_grouped_params(model, args), lr=args.learning_rate)
lr_scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.warmup_steps,
    num_training_steps=args.max_steps,
)
print(optimizer)
print(lr_scheduler)

accelerator.register_for_checkpointing(lr_scheduler)

def get_lr():
    return optimizer.param_groups[0]["lr"]

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

model.train()
completed_steps = 0
accumulated_loss_unlearn = 0.
accumulated_loss_mainrain = 0.
accumulated_loss_total = 0.

for step, batch in enumerate(tqdm(train_dataloader), start=1):
    outputs = model(input_ids=batch, labels=batch)
    loss_total = -outputs.loss
    #loss_total = outputs.loss # to check if the loss is calculated correctly
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

    if completed_steps >= args.max_steps:
        break

accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(model_dir, save_function=accelerator.save)
save_dir = model_dir / name
accelerator.save_state(save_dir)