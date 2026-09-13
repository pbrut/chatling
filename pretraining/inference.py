from models.gpt import GPT, GPTConfig
from utils.utils import get_latest_checkpoint

import torch
import tiktoken

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokeniser = tiktoken.get_encoding('gpt2')

checkpoint_path = get_latest_checkpoint('checkpoints/pretraining', 'model_step*.pth')
print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)

config = GPTConfig(**checkpoint['config'])
model = GPT(config=config)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

prompt = input("Prompt: ")
input = tokeniser.encode(prompt) # (T,)

input = torch.tensor(input, dtype=torch.long).unsqueeze(0) # (1, T)
input = input.to(device)

with torch.no_grad():
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        output = model.generate(input, max_new_tokens=1000, context_size=config.context_size) # (1, T)
    output = output[0] # (T,)
    output = output.tolist() # tiktoken expects a Python list

print(tokeniser.decode(output))
