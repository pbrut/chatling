from dataclasses import replace
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import os
import csv
import inspect
import torch
import torch.nn as nn

from models.gpt import GPT, GPTConfig
from utils.utils import ChatSFTDataset, get_latest_checkpoint, make_lr_lambda, save_checkpoint
from sft.extend_tokeniser import NUM_TOKENS


def load_pretrained_model(checkpoint_path, num_tokens):
    """
    Loads a pretraining checkpoint into a GPT sized for the larger chat
    vocab. Every weight is copied as-is except the tied token_embeddings /
    output table: its first `old_num_tokens` rows are copied in, and the
    extra rows for the new chat special tokens stay randomly initialised -
    those are exactly what SFT trains.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    old_config = GPTConfig(**checkpoint["config"])
    old_state = checkpoint["model_state_dict"]

    new_config = replace(old_config, num_tokens=num_tokens)
    model = GPT(config=new_config)

    tied_keys = ("transformer.token_embeddings.weight", "output.weight")
    filtered_state = {k: v for k, v in old_state.items() if k not in tied_keys}
    model.load_state_dict(filtered_state, strict=False)
    with torch.no_grad():
        model.transformer.token_embeddings.weight[:old_config.num_tokens] = old_state["transformer.token_embeddings.weight"]

    return model, new_config


# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
# uv run torchrun --standalone --nproc_per_node=<num_gpus> -m sft.finetune
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
    print(f"Running parallel SFT across {world_size} GPU(s)...")

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

batch_size = 16
epochs = 1
max_lr = 3e-5  # SFT uses a much lower LR than pretraining, and no big warmup/decay swing
min_lr = max_lr * 0.1
warmup_steps = 100
betas = (0.9, 0.95)
eps = 1e-8
weight_decay = 0.01
gsm8k_epochs = 4
global_steps = 0

pretrained_checkpoint = get_latest_checkpoint('checkpoints/pretraining', 'model_step*.pth')
if master_process:
    print(f"Loading pretrained checkpoint: {pretrained_checkpoint}")
    
model, config = load_pretrained_model(pretrained_checkpoint, NUM_TOKENS)
model.to(device)
model = torch.compile(model=model)
model = DDP(module=model, device_ids=[local_rank])

train_dataset = ChatSFTDataset(split="train", context_size=config.context_size, gsm8k_epochs=gsm8k_epochs)
# We can use a Sampler here to distribute the data across ranks because the dataset is very small.
# The reason we could not use it for fineweb is because we had to load the entire dataset to memory, which
# was not possible, but is possible with smoltalk.
train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=1337)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True)

val_dataset = ChatSFTDataset(split="test", context_size=config.context_size, gsm8k_epochs=gsm8k_epochs)
val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, sampler=val_sampler, pin_memory=True)

step_tokens = 131072  # ~2**17 tokens per optimizer step; smaller than pretraining since SFT data is scarce
batch_tokens = batch_size * config.context_size * world_size
assert step_tokens % batch_tokens == 0
num_batches_per_step = step_tokens // batch_tokens
ready_to_take_step = lambda batch: (batch % num_batches_per_step == 0)
num_steps_to_eval = 50
should_evaluate = lambda step: (step % num_steps_to_eval == 0)
num_steps_to_checkpoint = num_steps_to_eval * 2
should_checkpoint = lambda step: (step % num_steps_to_checkpoint == 0)

if master_process:
    print(f"Gradient will be accumulated over {num_batches_per_step * world_size} batches across {world_size} GPU(s).")

num_batches = len(train_loader)
num_steps_per_epoch = num_batches // num_batches_per_step
max_steps = num_steps_per_epoch * epochs

use_fused_adam = 'fused' in inspect.signature(AdamW).parameters and 'cuda' in device
print(f"[{device}] Using fused AdamW: {use_fused_adam}")
optimizer = AdamW(model.parameters(), lr=max_lr, betas=betas, eps=eps, weight_decay=weight_decay, fused=use_fused_adam)
lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=make_lr_lambda(warmup_steps, max_steps, min_lr, max_lr))

if master_process:
    os.makedirs('results/sft', exist_ok=True)
    os.makedirs('checkpoints/sft', exist_ok=True)
    log_file = open('results/sft/log.csv', 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['step', 'split', 'value'])
    log_file.flush()

for epoch in range(epochs):
    train_sampler.set_epoch(epoch)

    epoch_loss = 0.0
    step_loss = 0.0
    steps = 0
    for i, (xb, yb) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}"), start=1):
        xb = xb.to(device)
        yb = yb.to(device)

        # For the batch right before we take a step, both the forward and backward passes
        # need the 'model.require_backward_grad_sync' set to 'True'.
        model.require_backward_grad_sync = ready_to_take_step(i)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, batch_loss = model(xb, yb)

        batch_loss = batch_loss / num_batches_per_step
        step_loss += batch_loss

        # Backward pass on the gradient accumulated over the last num_batches_per_step batches
        batch_loss.backward()

        if ready_to_take_step(i):
            # Average of all 'step_loss' values from all ranks, and store it on all ranks.
            torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)

            loss = step_loss.item()
            epoch_loss += step_loss / num_steps_per_epoch
            step_loss = 0.0
            steps += 1

            # Clip the gradient magnitude to below 1.0, to avoid an exploading gradient
            l2norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if master_process:
                tqdm.write(f"Step: {global_steps + steps} | Loss: {loss:.5f} | L2 norm: {l2norm:.4f} | Lr: {lr_scheduler.get_last_lr()[0]:.4e}")
                log_writer.writerow([global_steps + steps, 'train', loss])
                log_file.flush()

            if should_evaluate(steps):
                model.eval()

                with torch.no_grad():
                    val_loss = 0.0
                    stop_validation = False

                    # Every val_loss is calculated on the same examples
                    for j, (xb, yb) in enumerate(tqdm(val_loader, desc=f"Evaluation at step {steps}", total=num_batches_per_step), start=1):
                        if stop_validation:
                            break

                        xb = xb.to(device)
                        yb = yb.to(device)

                        with torch.autocast(device_type=device, dtype=torch.bfloat16):
                            _, val_batch_loss = model(xb, yb)

                        val_loss += val_batch_loss / num_batches_per_step

                        # We evaluate over the first num_batches_per_step batches
                        if ready_to_take_step(j):
                            stop_validation = True

                    torch.distributed.all_reduce(val_loss, op=torch.distributed.ReduceOp.AVG)

                    if master_process:
                        print(f"Validation loss at step {global_steps + steps}: {val_loss.item():.5f}\n")
                        log_writer.writerow([global_steps + steps, 'val', val_loss.item()])
                        log_file.flush()

                    if master_process and should_checkpoint(steps):
                        raw_model = model.module._orig_mod
                        checkpoint_path = f'checkpoints/sft/model_step{global_steps + steps}.pth'
                        save_checkpoint(
                            checkpoint_path, raw_model, config, optimizer, lr_scheduler,
                            epoch=epoch, step=steps, global_step=global_steps + steps,
                            train_loss=loss, val_loss=val_loss.item(),
                        )
                        tqdm.write(f"Checkpoint saved to {checkpoint_path}")

                model.train()

    if master_process:
        print(f"Epoch {epoch} | Loss {(epoch_loss):.5f}")
        global_steps += steps

if master_process:
    # The flow is:
    #  1. torch.compile stores the original model at 'model._orig_mod'
    #  2. DDP stores the compiled model at 'model.module'

    raw_model = model.module._orig_mod
    checkpoint_path = f'checkpoints/sft/model_step{global_steps}.pth'
    save_checkpoint(
        checkpoint_path, raw_model, config, optimizer, lr_scheduler,
        epoch=epoch, step=steps, global_step=global_steps,
        train_loss=loss, val_loss=val_loss.item() if val_loss is not None else None,
    )
    print(f"Checkpoint saved to {checkpoint_path}")
    log_file.close()

if is_ddp:
    destroy_process_group()
