from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset
from tqdm import tqdm

import os
import csv
import re
import inspect
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gpt import GPT, GPTConfig, IGNORE_TOKEN
from utils.utils import get_latest_checkpoint, make_lr_lambda, save_checkpoint
from sft.extend_tokeniser import (
    tokeniser, EOT_TOKEN, USER_START_TOKEN, USER_END_TOKEN, ASSISTANT_START_TOKEN, ASSISTANT_END_TOKEN,
)


# Binary reward, if answer matches exactly, reward=1, otherwise, reward=0. 
GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def extract_answer(text):
    match = GSM_RE.search(text)
    if match is None:
        return None
    return match.group(1).strip().replace(",", "")


def compute_reward(generated_text, ground_truth_answer):
    return float(extract_answer(generated_text) == extract_answer(ground_truth_answer))


def generate_prompt(question):
    return [EOT_TOKEN, USER_START_TOKEN] + tokeniser.encode_ordinary(question) + [USER_END_TOKEN, ASSISTANT_START_TOKEN]


def decode_response(sequence, prompt_len):
    # Only decode up to the first <|assistant_end|>; anything the model rambles on
    # with after that is not part of "the answer" and shouldn't affect the reward.
    response_ids = sequence[prompt_len:].tolist()
    if ASSISTANT_END_TOKEN in response_ids:
        response_ids = response_ids[:response_ids.index(ASSISTANT_END_TOKEN)]
    return tokeniser.decode(response_ids)


def build_rollout_mask(sequence, prompt_len, total_len):
    # Mask out anything that isn't the model generated answer ending with <|assistant_end|>.
    end_pos = total_len - 1
    for t in range(prompt_len, total_len):
        if sequence[t].item() == ASSISTANT_END_TOKEN:
            end_pos = t
            break
    mask = torch.zeros(total_len - 1, dtype=torch.long)
    mask[prompt_len - 1:end_pos] = 1
    return mask


def load_sft_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = GPTConfig(**checkpoint["config"])
    model = GPT(config=config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, config


# -----------------------------------------------------------------------------
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
# uv run torchrun --standalone --nproc_per_node=<num_gpus> -m rlvr.finetune
is_ddp = int(os.environ.get('RANK', -1)) != -1

assert is_ddp, "Run this script via torchrun"
assert torch.cuda.is_available(), "CUDA is required for DDP"

init_process_group(backend='nccl')
rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])

device = f'cuda:{local_rank}'
torch.cuda.set_device(device)

master_process = rank == 0
if master_process:
    print(f"Running parallel RLVR across {world_size} GPU(s)...")

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

num_rollouts = 8
num_tokens_per_rollout = 200
num_steps = 2000 # one GSM8K question per step
max_lr = 5e-6
min_lr = max_lr * 0.1
warmup_steps = 20
betas = (0.9, 0.95)
eps = 1e-8
weight_decay = 0.0 # no regularisation
num_steps_to_eval = 50
num_eval_examples = 100
num_steps_to_checkpoint = 100

sft_checkpoint = get_latest_checkpoint('checkpoints/sft', 'model_step*.pth')
if master_process:
    print(f"Loading SFT checkpoint: {sft_checkpoint}")

# Using the 'raw_model' for .generate() because running it on a compiled model can cause issues.
raw_model, config = load_sft_model(sft_checkpoint)
raw_model.to(device)
model = torch.compile(model=raw_model)
model = DDP(module=model, device_ids=[local_rank])

train_dataset = load_dataset("openai/gsm8k", "main", split="train")
test_dataset = load_dataset("openai/gsm8k", "main", split="test")
rank_indices = range(rank, len(train_dataset), world_size)

# Infinite iterator that starts from the beginning if it reaches the end of the range.
rank_questions = itertools.cycle(rank_indices)

use_fused_adam = 'fused' in inspect.signature(AdamW).parameters and 'cuda' in device
print(f"[{device}] Using fused AdamW: {use_fused_adam}")
optimizer = AdamW(model.parameters(), lr=max_lr, betas=betas, eps=eps, weight_decay=weight_decay, fused=use_fused_adam)
lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=make_lr_lambda(warmup_steps, num_steps, min_lr, max_lr))

if master_process:
    os.makedirs('results/rlvr', exist_ok=True)
    os.makedirs('checkpoints/rlvr', exist_ok=True)
    log_file = open('results/rlvr/log.csv', 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['step', 'split', 'value'])
    log_file.flush()

for step in range(1, num_steps + 1):
    question_idx = next(rank_questions)
    row = train_dataset[question_idx]
    prompt = generate_prompt(row["question"])
    prompt_len = len(prompt)

    # Generate 'num_rollouts' independent completions of the same prompt.
    raw_model.eval()
    input = torch.tensor([prompt] * num_rollouts, dtype=torch.long, device=device) # (num_rollouts, prompt_len)
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            sequences = raw_model.generate(input, max_new_tokens=num_tokens_per_rollout, context_size=config.context_size) # (num_rollouts, num_tokens_per_rollout)

    # E.g. rewards = [1, 1, 0, 0, ...]  
    rewards = torch.tensor(
        [compute_reward(decode_response(sequences[i], prompt_len), row["answer"]) for i in range(num_rollouts)],
        dtype=torch.float, device=device
    )

    # GRPO-style (no KL divergence) advantage
    advantages = rewards - rewards.mean()

    masks = torch.stack([
        build_rollout_mask(sequences[i], prompt_len, sequences.size(1)) for i in range(num_rollouts)
    ]).to(device)

    inputs = sequences[:, :-1]
    targets = sequences[:, 1:].clone()
    targets[masks == 0] = IGNORE_TOKEN

    raw_model.train()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, _ = model(inputs)

    B, T, C = logits.shape

    # The losses for individual tokens from a single rollout are scaled by the corresponding advantage. 
    nll = F.cross_entropy(logits.reshape(B * T, C), targets.reshape(B * T), ignore_index=IGNORE_TOKEN, reduction='none')
    nll = nll.view(B, T)
    advantages = advantages.unsqueeze(-1) # (B, 1)
    num_valid = (targets != IGNORE_TOKEN).sum().clamp(min=1)
    loss = (nll * advantages).sum() / num_valid

    loss.backward()

    l2norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    mean_reward = rewards.mean()
    torch.distributed.all_reduce(mean_reward, op=torch.distributed.ReduceOp.AVG)

    if master_process:
        tqdm.write(
            f"Step: {step} | Loss: {loss.item():.5f} | L2 norm: {l2norm:.4f} "
            f"| Lr: {lr_scheduler.get_last_lr()[0]:.4e} | Avg reward: {mean_reward.item():.4f}"
        )
        log_writer.writerow([step, 'train_loss', loss.item()])
        log_writer.writerow([step, 'train_reward', mean_reward.item()])
        log_file.flush()

    # -------------------------------------------------------------------------
    if step % num_steps_to_eval == 0 or step == num_steps:
        raw_model.eval()
        num_correct = 0
        num_total = 0
        for idx in range(rank, min(num_eval_examples, len(test_dataset)), world_size):
            row = test_dataset[idx]
            prompt = generate_prompt(row["question"])
            input = torch.tensor([prompt], dtype=torch.long, device=device)
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    sequence = raw_model.generate(input, max_new_tokens=num_tokens_per_rollout, context_size=config.context_size)
            response = decode_response(sequence[0], len(prompt))
            num_correct += int(compute_reward(response, row["answer"]) == 1.0)
            num_total += 1
        raw_model.train()

        num_correct = torch.tensor(num_correct, dtype=torch.long, device=device)
        num_total = torch.tensor(num_total, dtype=torch.long, device=device)
        torch.distributed.all_reduce(num_correct, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(num_total, op=torch.distributed.ReduceOp.SUM)

        if master_process:
            accuracy = num_correct.item() / num_total.item()
            print(f"GSM8K pass@1 at step {step}: {num_correct.item()}/{num_total.item()}={accuracy:.4f}\n")
            log_writer.writerow([step, 'gsm8k_accuracy', accuracy])
            log_file.flush()

    if master_process and (step % num_steps_to_checkpoint == 0 or step == num_steps):
        checkpoint_path = f'checkpoints/rlvr/model_step{step}.pth'
        save_checkpoint(
            checkpoint_path, raw_model, config, optimizer, lr_scheduler,
            epoch=0, step=step, global_step=step,
            train_loss=loss.item(),
        )
        tqdm.write(f"Checkpoint saved to {checkpoint_path}")

if master_process:
    log_file.close()

if is_ddp:
    destroy_process_group()
