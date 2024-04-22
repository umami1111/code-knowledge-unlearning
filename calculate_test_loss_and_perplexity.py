import torch
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe
from datasets import Dataset
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


device = "cuda:0"
src_dir = "Source/source_mit_python_2023-03-28-14-47-19"
batch_size = 8
seq_length = 1024
shuffle_buffer = 10000

# Rewrite these for each model
# default codeparrot-small
tokenizer = AutoTokenizer.from_pretrained("codeparrot/codeparrot-small")
model = AutoModelForCausalLM.from_pretrained("codeparrot/codeparrot-small").to(device)
# unlearn model
# model_dir = "output/unlearning/models/lr-5e-05_bs-2_accsteps-1_epochs-1.0_maxsteps-0_warmsteps-0_lora-8_seed-42"
# tokenizer = AutoTokenizer.from_pretrained(model_dir)
# model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)


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
                    else:
                        more_examples = False
                        break
            if self.tokenized:
                tokenized_inputs = buffer
            else:
                tokenized_inputs = self.tokenizer(buffer, padding=False, truncation=False)["input_ids"]
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


def create_dataloader(src_dir, tokenizer, seq_length, shuffle_buffer, batch_size):
    src_dir = Path(src_dir)
    data = []
    for p in src_dir.glob("*.py"):
        with open(p, "r", encoding="utf-8") as f:
            data.append({"content": f.read()})

    dataset = Dataset.from_list(data)
    print(dataset)
    print(dataset[0])
    dataset = ConstantLengthDataset(
        tokenizer, dataset, infinite=True, seq_length=seq_length, tokenized=False
    )
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


def evaluate(model, dataloader, eval_steps, device):
    model.eval()
    losses = []
    for step, batch in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            batch = batch.to(device)
            outputs = model(batch, labels=batch)
        loss = outputs.loss.repeat(batch_size)
        losses.append(loss)
        if step >= eval_steps:
            break
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()


dataloader = create_dataloader(src_dir, tokenizer, seq_length, shuffle_buffer, batch_size)
eval_steps = 2000//batch_size
loss, ppl = evaluate(model, dataloader, eval_steps, device)
print(f"loss: {loss}, perplexity: {ppl}")