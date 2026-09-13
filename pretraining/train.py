from utils.utils import HellaSwagDataset, ShardIterableDataset, make_lr_lambda, save_checkpoint
from models.gpt import GPT, GPTConfig
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import os
import csv
import glob
import inspect
import torch
import torch.nn as nn


# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
# uv run torchrun --standalone --nproc_per_node=1 -m train
is_ddp = int(os.environ.get('RANK', -1)) != -1

assert is_ddp, "Run this script via torchrun"
assert torch.cuda.is_available(), "CUDA is required for DDP"

init_process_group(backend='nccl')
rank = int(os.environ['RANK']) # global ID for each GPU across all machines
local_rank = int(os.environ['LOCAL_RANK']) # local ID for a GPU on a single machine
world_size = int(os.environ['WORLD_SIZE']) # number of GPUs across all machines

device = f'cuda:{local_rank}'
torch.cuda.set_device(device)

# This process will do logging, checkpointing etc.
master_process = rank == 0
if master_process:
    print(f"Running parallel training across {world_size} GPU(s)...") 


# Each GPU initialises the same weights
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

config = GPTConfig()
batch_size = 32

shard_paths = sorted(glob.glob("data/edu_fineweb10B/edufineweb_train_*.npy"))
dataset = ShardIterableDataset(shard_paths=shard_paths, context_length=config.context_size, rank=rank, world_size=world_size)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size,
    shuffle=False, # Needs to be False if we use an iterable dataset
    pin_memory=True # Speeds up the data transfer between the CPU and GPU
)

eval_shard_paths = sorted(glob.glob("data/edu_fineweb10B/edufineweb_val_*.npy"))
eval_dataset = ShardIterableDataset(shard_paths=eval_shard_paths, context_length=config.context_size, rank=rank, world_size=world_size)
eval_data_loader = DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

hellaswag_dataset = HellaSwagDataset(split="val", rank=rank, world_size=world_size)

# GPT-2 124M used 2**19 = ~0.5M tokens per optimizer step
step_tokens = 524288 # tokens per step across all GPUs
batch_tokens = batch_size * config.context_size * world_size
assert step_tokens % batch_tokens == 0
num_batches_per_step = step_tokens // batch_tokens # Each GPU will process 'num_batches_per_step' batches
ready_to_take_step = lambda batch: (batch % num_batches_per_step == 0)
num_steps_to_eval = 100
should_evaluate = lambda step: (step % num_steps_to_eval == 0)
num_steps_to_checkpoint = num_steps_to_eval * 20
should_checkpoint = lambda step: (step % num_steps_to_checkpoint == 0)

epochs = 1
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
betas = (0.9, 0.95)
eps = 1e-8
weight_decay = 0.1
# For plotting step losses across epochs
global_steps = 0

if master_process:
    print(f"Gradient will be accumulated over {num_batches_per_step * world_size} batches across {world_size} GPU(s).")

model = GPT(config=config)
model.to(device)
model = torch.compile(model=model)
model = DDP(module=model, device_ids=[local_rank])

num_batches = len(data_loader)
num_steps_per_epoch = num_batches // num_batches_per_step
max_steps = num_steps_per_epoch * epochs

use_fused_adam = 'fused' in inspect.signature(AdamW).parameters and 'cuda' in device
print(f"[{device}] Using fused AdamW: {use_fused_adam}")
optimizer = AdamW(model.parameters(), lr=max_lr, betas=betas, eps=eps, weight_decay=weight_decay, fused=use_fused_adam)
lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=make_lr_lambda(warmup_steps, max_steps, min_lr, max_lr))

if master_process:
    os.makedirs('results/pretraining', exist_ok=True)
    os.makedirs('checkpoints/pretraining', exist_ok=True)
    log_file = open('results/pretraining/log.csv', 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['step', 'split', 'value'])
    log_file.flush()

for epoch in range(epochs):
    # Shuffle dataset for each epoch
    dataset.set_epoch(epoch=epoch)

    epoch_loss = 0.0
    step_loss = 0.0
    steps = 0
    for i, (xb, yb) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}"), start=1):
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
                    # =================================
                    # RUN FINEWEB EVAL
                    # =================================
                    val_loss = 0.0
                    stop_validation = False

                    # Every val_loss is calculated on the same examples
                    for j, (xb, yb) in enumerate(tqdm(eval_data_loader, desc=f"Evaluation at step {steps}", total=num_batches_per_step), start=1):
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

                    # =================================
                    # RUN HELLASWAG EVAL    
                    # =================================
                    num_correct = 0
                    num_total = 0
                    for num_total, (tokens, mask, label) in enumerate(tqdm(hellaswag_dataset, desc=f"HellaSwag eval at step {steps}", total=len(hellaswag_dataset)), start=1):
                        tokens = tokens.to(device)
                        mask = mask.to(device)

                        with torch.autocast(device_type=device, dtype=torch.bfloat16):
                            logits, _ = model(tokens)

                        pred = hellaswag_dataset.get_most_likely_row(tokens, mask, logits)
                        num_correct += int(pred == label)

                    num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                    num_correct = torch.tensor(num_correct, dtype=torch.long, device=device)
                    torch.distributed.all_reduce(num_total, op=torch.distributed.ReduceOp.SUM)
                    torch.distributed.all_reduce(num_correct, op=torch.distributed.ReduceOp.SUM)

                    if master_process:
                        num_correct = num_correct.item()
                        num_total = num_total.item()
                        accuracy = num_correct / num_total
                        print(f"HellaSwag accuracy at step {global_steps + steps}: {num_correct}/{num_total}={accuracy:.4f}\n")
                        log_writer.writerow([global_steps + steps, 'hellaswag', accuracy])
                        log_file.flush()

                    if master_process and should_checkpoint(steps):
                        # To resume training after a full epoch, need to add some changes to support the epoch += 1,
                        # and need to figure out how to handle the learning rate, because max_steps is dependent on the
                        # number of epochs.
                        raw_model = model.module._orig_mod
                        checkpoint_path = f'checkpoints/pretraining/model_step{global_steps + steps}.pth'
                        save_checkpoint(
                            checkpoint_path, raw_model, config, optimizer, lr_scheduler,
                            epoch=epoch, step=steps, global_step=global_steps + steps,
                            train_loss=loss, val_loss=val_loss.item(), hellaswag_accuracy=accuracy,
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
    checkpoint_path = f'checkpoints/pretraining/model_step{global_steps}.pth'
    save_checkpoint(
        checkpoint_path, raw_model, config, optimizer, lr_scheduler,
        epoch=epoch, step=steps, global_step=global_steps,
        train_loss=loss, val_loss=val_loss.item() if val_loss is not None else None, hellaswag_accuracy=accuracy,
    )
    print(f"Checkpoint saved to {checkpoint_path}")
    log_file.close()
    
if is_ddp:
    destroy_process_group()