import os
import requests
from tqdm import tqdm

URLS = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

DATA_CACHE_DIR = os.path.join("data", "hellaswag")


def data_path(split):
    return os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")


def download(split, chunk_size=1024):
    assert split in URLS, f"split must be one of {list(URLS)}"
    path = data_path(split)
    if os.path.exists(path):
        return path

    print(f"Downloading hellaswag_{split}.jsonl...")
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    resp = requests.get(URLS[split], stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(path, "wb") as file, tqdm(
        desc=path,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

    print(f"Downloaded {path}")
    return path


if __name__ == "__main__":
    download("val")
