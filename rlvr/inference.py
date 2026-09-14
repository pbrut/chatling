from models.gpt import GPT, GPTConfig
from utils.utils import get_latest_checkpoint
from sft.extend_tokeniser import (
    tokeniser, EOT_TOKEN, USER_START_TOKEN, USER_END_TOKEN, ASSISTANT_START_TOKEN, ASSISTANT_END_TOKEN,
)

import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_path = get_latest_checkpoint('checkpoints/rlvr', 'model_step*.pth')
print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)

config = GPTConfig(**checkpoint['config'])
model = GPT(config=config)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

max_new_tokens = 1000
history = [EOT_TOKEN]

while True:
    prompt = input("You: ")
    if not prompt:
        break

    history += [USER_START_TOKEN] + tokeniser.encode_ordinary(prompt) + [USER_END_TOKEN, ASSISTANT_START_TOKEN]

    input_ids = torch.tensor(history, dtype=torch.long).unsqueeze(0).to(device) # (1, T)

    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            output = model.generate(input_ids, max_new_tokens=max_new_tokens, context_size=config.context_size) # (1, T)

    response = output[0, len(history):].tolist()
    if ASSISTANT_END_TOKEN in response:
        response = response[:response.index(ASSISTANT_END_TOKEN)]

    print(f"Assistant: {tokeniser.decode(response)}\n")

    history += response + [ASSISTANT_END_TOKEN]
