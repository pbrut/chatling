import glob
import json
import math
import os
import random
import tiktoken
import torch
import torch.nn.functional as F
import numpy as np

from dataclasses import asdict
from datasets import load_dataset
from torch.utils.data import Dataset, IterableDataset

from utils import hellaswag
from models.gpt import IGNORE_TOKEN
from sft.extend_tokeniser import (
    tokeniser, EOT_TOKEN, USER_START_TOKEN, USER_END_TOKEN, ASSISTANT_START_TOKEN, ASSISTANT_END_TOKEN,
)


def get_latest_checkpoint(checkpoint_dir, pattern):
    checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, pattern))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints matching '{pattern}' found in '{checkpoint_dir}'")
    prefix, suffix = pattern.split('*')
    return max(checkpoint_paths, key=lambda path: int(os.path.basename(path)[len(prefix):-len(suffix)]))


# LR schedule: linear warmup from ~0 to max_lr over warmup_steps, then cosine decay
# to min_lr by max_steps, then constant at min_lr. Returns a multiplier (0–1) because
# LambdaLR scales the optimizer's base lr by this value.
def make_lr_lambda(warmup_steps, max_steps, min_lr, max_lr):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        if step > max_steps:
            return min_lr / max_lr
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return (min_lr + coeff * (max_lr - min_lr)) / max_lr
    return lr_lambda


# The 'model' must already be unwrapped from torch.compile/DDP 
def save_checkpoint(checkpoint_path, model, config, optimizer, scheduler, epoch, step, global_step,
                     train_loss=None, val_loss=None, hellaswag_accuracy=None):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': asdict(config),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'step': step,
        'global_step': global_step,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'hellaswag_accuracy': hellaswag_accuracy,
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(checkpoint, checkpoint_path)


# Divide the dataset into 'world_size' sub-datasets,
# each sub-dataset will be processed by a different GPU in parallel.
class ShardIterableDataset(IterableDataset):
    def __init__(self, shard_paths, context_length, rank, world_size):
        self.shard_paths = shard_paths
        self.context_length = context_length
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _get_trimmed_indices(self, num_tokens):
        # Create NON-OVERLAPPING sequence starts
        # If context_length=1024, using step=self.context_length generates: 0, 1024, 2048, 3072...
        valid_indices = np.arange(0, num_tokens - self.context_length, self.context_length, dtype=np.uint32)

        # Throw away the remainder if shard is not divisible by world_size
        remainder = len(valid_indices) % self.world_size
        if remainder > 0:
            valid_indices = valid_indices[:-remainder]

        return valid_indices

    def __iter__(self):
        # Keeping this in case I want to do num_workers > 0 for the CPU in the DataLoader
        worker_info = torch.utils.data.get_worker_info()
        # Unique seed for each rank and each epoch
        seed = self.rank + self.epoch + (worker_info.id if worker_info else 0)
        rng = np.random.default_rng(seed)
        
        shards = list(self.shard_paths)
        rng.shuffle(shards)

        for shard_path in shards:
            tokens = np.load(shard_path, mmap_mode="r")
            
            valid_indices = self._get_trimmed_indices(len(tokens))
            
            # Divide into contiguous chunks for each GPU
            chunks = np.array_split(valid_indices, self.world_size)
            rank_indices = chunks[self.rank]

            # Shuffle this GPU's dedicated chunk so batches are randomized
            rng.shuffle(rank_indices)

            for idx in rank_indices:
                x = tokens[idx : idx + self.context_length].astype(np.int64)
                y = tokens[idx + 1 : idx + self.context_length + 1].astype(np.int64)
                yield torch.from_numpy(x), torch.from_numpy(y)

    def __len__(self):
        total_samples = 0
        for p in self.shard_paths:
            tokens = np.load(p, mmap_mode="r")
            trimmed = self._get_trimmed_indices(len(tokens))
            total_samples += len(trimmed)

        return total_samples // self.world_size


class ChatSFTDataset(Dataset):
    """
    Mixture of SmolTalk (general conversation) and GSM8K (math
    reasoning) into fixed-length (context_size + 1) token/mask rows.

    GSM8K's train split (~7.5K rows) is tiny next to SmolTalk's (~460K rows),
    so it's repeated 'gsm8k_epochs' times. The two datasets are mixed so each 
    batch should contain examples from both datasets. 
    """

    def __init__(self, split, context_size, gsm8k_epochs=4):
        self.smoltalk = load_dataset("HuggingFaceTB/smol-smoltalk", split=split)
        self.gsm8k = load_dataset("openai/gsm8k", "main", split=split)
        self.context_size = context_size

        gsm8k_repeats = gsm8k_epochs if split == "train" else 1
        index_map = [("smoltalk", i) for i in range(len(self.smoltalk))]
        index_map += [("gsm8k", i) for _ in range(gsm8k_repeats) for i in range(len(self.gsm8k))]
        random.Random(1337).shuffle(index_map)
        self.index_map = index_map

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        source, local_idx = self.index_map[idx]
        if source == "smoltalk":
            conversation = list(self.smoltalk[local_idx]["messages"])
        else:
            row = self.gsm8k[local_idx]
            conversation = [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]},
            ]

        if conversation and conversation[0]["role"] == "system":
            # If the first message is a 'system' message (e.g. You are an AI system tasked with...), add that to the 'user' message
            assert len(conversation) >= 2 and conversation[1]["role"] == "user"
            conversation[1] = {"role": "user", "content": conversation[0]["content"] + "\n\n" + conversation[1]["content"]}
            conversation = conversation[1:]

        # Tokenise the conversation. Set mask=1 for tokens the model is trained to
        # predict (i.e. assistant generated answer and its end <|assistant_end|>), mask=0
        # for everything else (user turns, role tokens, padding).
        tokens = [EOT_TOKEN]
        mask = [0]
        for i, message in enumerate(conversation):
            # We expect an alternating series of 'user', 'assistant', 'user', 'assistant'... 
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected_role, f"message {i} has role {message['role']}, expected {expected_role}"
            message_tokens = tokeniser.encode_ordinary(message["content"])

            if message["role"] == "user":
                tokens += [USER_START_TOKEN] + message_tokens + [USER_END_TOKEN]
                mask += [0] * (len(message_tokens) + 2)
            else:
                tokens += [ASSISTANT_START_TOKEN] + message_tokens + [ASSISTANT_END_TOKEN]
                mask += [0] + [1] * len(message_tokens) + [1]

        row_capacity = self.context_size + 1
        tokens = tokens[:row_capacity]
        mask = mask[:row_capacity]

        # Pad if conversation is shorter than the context_size
        pad = row_capacity - len(tokens)
        if pad > 0:
            tokens += [EOT_TOKEN] * pad
            mask += [0] * pad

        tokens = torch.tensor(tokens, dtype=torch.long)
        mask = torch.tensor(mask, dtype=torch.long)

        x = tokens[:-1]
        y = tokens[1:].clone()
        y[mask[1:] == 0] = IGNORE_TOKEN

        return x, y


class HellaSwagDataset(IterableDataset):

    def __init__(self, split, rank=0, world_size=1):
        assert split in hellaswag.URLS, f"split must be one of {list(hellaswag.URLS)}"
        self.split = split
        self.rank = rank
        self.world_size = world_size
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.data_path = hellaswag.data_path(split)
        assert os.path.exists(self.data_path), (
            f"'{self.data_path}' not found, run 'uv run python -m utils.hellaswag' first"
        )

    def _render_example(self, example):
        """
        Given an example dict, render it as:
        - tokens: (4, N) - context + candidate ending tokens, one row per ending
        - mask: (4, N) - 1 over the candidate ending region, where we evaluate likelihoods
        - label: index of the correct ending
        """
        ctx_tokens = self.tokenizer.encode(example["ctx"])
        label = example["label"]

        tok_rows = []
        mask_rows = []
        for end in example["endings"]:
            end_tokens = self.tokenizer.encode(" " + end)
            tok_rows.append(ctx_tokens + end_tokens)
            mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))

        max_len = max(len(row) for row in tok_rows)
        tokens = torch.zeros((4, max_len), dtype=torch.long)
        mask = torch.zeros((4, max_len), dtype=torch.long)
        for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
            tokens[i, :len(tok_row)] = torch.tensor(tok_row)
            mask[i, :len(mask_row)] = torch.tensor(mask_row)

        return tokens, mask, label

    def __iter__(self):
        with open(self.data_path, "r") as f:
            # We don't have to worry about number of examples being divisible by world_size.
            for i, line in enumerate(f):
                if i % self.world_size != self.rank:
                    continue
                yield self._render_example(json.loads(line))

    def __len__(self):
        with open(self.data_path, "r") as f:
            total = sum(1 for _ in f)
        return total // self.world_size + (1 if self.rank < total % self.world_size else 0)

    def get_most_likely_row(self, tokens, mask, logits):
        # Throw away the last (1, num_tokens) row, we won't be predicting next token
        logits = logits[..., :-1, :] # (B, context_size - 1, num_tokens)
        # Throw away the first token, nothing predicts it
        tokens = tokens[..., 1:] # (B, context_size - 1)
        # Throw away the first mask
        mask = mask[..., 1:] # (B, context_size - 1)

        B, T, C = logits.shape

        logits = logits.reshape(B*T, C)
        tokens = tokens.reshape(B*T)

        losses = F.cross_entropy(logits, tokens, reduction='none') # (B*T,)
        losses = losses.reshape(B, T) # (B, T)
        
        masked_losses = losses * mask # (B, T)

        # Calculate average loss from each row
        mean_losses = masked_losses.sum(dim=-1) / mask.sum(dim=-1)
        
        # Example with the lowest loss is the most likely
        pred = mean_losses.argmin().item()
        return pred
