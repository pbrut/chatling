import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset

local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8) # 100M tokens per shard, total of 100 shards
num_shards_to_download = None # set to an int to download a custom number of shards

DATA_CACHE_DIR = os.path.join("data", local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

dataset = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train", streaming=True)

tokeniser = tiktoken.get_encoding("gpt2")
eot = tokeniser._special_tokens['<|endoftext|>'] # special end of text token, <|endoftext|>


def tokenize(doc):
    tokens = [eot] # delimits all documents
    tokens.extend(tokeniser.encode_ordinary(doc["text"]))
    tokens = np.array(tokens)

    # Because GPT-2 tokeniser only uses 50257 tokens, we do not need more than 2**16 integers,
    # so to save space, we can make all tokens uint16.
    assert (0 <= tokens).all() and (tokens < 2**16).all(), "Token dictionary too large for uint16"
    tokens = tokens.astype(np.uint16)
    return tokens


def save_shard(filename, tokens_np):
    np.save(filename, tokens_np)
    print(f"Saved shard: {filename}")


# Number of independent processes.
num_processes = max(1, os.cpu_count() // 2)

with mp.Pool(num_processes) as pool:
    
    shard_index = 0
    shard = np.empty((shard_size,), dtype=np.uint16)
    shard_tokens = 0
    print(f"Downloading shard {shard_index}...")

    for tokens in pool.imap(tokenize, dataset, chunksize=16):

        # Is there enough space in the current shard for the new tokens
        if shard_tokens + len(tokens) <= shard_size:
            shard[shard_tokens:shard_tokens + len(tokens)] = tokens
            shard_tokens += len(tokens)
        else:
            # Fill out the current shard
            remaining_space = shard_size - shard_tokens
            shard[shard_tokens:shard_tokens+remaining_space] = tokens[:remaining_space]

            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            save_shard(filename, shard)
            shard_index += 1

            if num_shards_to_download is not None and shard_index >= num_shards_to_download:
                break

            print(f"Downloading shard {shard_index}...")

            # Populate the next shard with the leftovers of the current doc
            remainder = len(tokens) - remaining_space
            shard[0:remainder] = tokens[remaining_space:]
            shard_tokens = remainder

    # Write any remaining tokens as the last shard
    if shard_tokens != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        save_shard(filename, shard[:shard_tokens])