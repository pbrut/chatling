from datasets import load_dataset


print("Downloading HuggingFaceTB/smol-smoltalk...")
load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
load_dataset("HuggingFaceTB/smol-smoltalk", split="test")

print("Downloading openai/gsm8k...")
load_dataset("openai/gsm8k", "main", split="train")
load_dataset("openai/gsm8k", "main", split="test")

print("Done.")
