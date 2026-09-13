from dataclasses import dataclass
from torch.nn import functional as F

import torch
import torch.nn as nn


# Used only during the SFT finetuning stage to mask out the 'user' parts of the conversation.
# The model should only learn how to predict the 'assistant' generated answers.
IGNORE_TOKEN = -1


@dataclass
class GPTConfig:
    token_embed: int = 768
    num_heads: int = 6
    num_layers: int = 12
    context_size: int = 2048  # GPT-3 size
    num_tokens: int = 50257 # number of tokens from the tiktoken GPT-2 tokeniser
    attention_bias: bool = False

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.token_embed % config.num_heads == 0

        self.token_embed = config.token_embed
        self.num_heads = config.num_heads
        self.head_size = config.token_embed // config.num_heads 

        self.qkv = nn.Linear(config.token_embed, 3 * config.token_embed, bias=config.attention_bias)

        self.output = nn.Linear(config.token_embed, config.token_embed, bias=config.attention_bias)
        self.output.is_residual = True

    def forward(self, x):
        B,T,C = x.shape

        qkv = self.qkv(x) # (B, T, 3 * token_embed)
        q, k, v = qkv.split(self.token_embed, dim=-1) # (B, T, token_embed)

        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2) # (B, num_heads, T, head_size)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2) # (B, num_heads, T, head_size)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2) # (B, num_heads, T, head_size)

        # Flash attention
        out = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=True) # (B, num_heads, T, head_size)
        out = out.transpose(1, 2).contiguous().view(B, T, C) # (B, T, token_embed)

        out = self.output(out) # (B, T, token_embed)

        return out


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.token_embed, 4 * config.token_embed),
            nn.GELU(),
            nn.Linear(4 * config.token_embed, config.token_embed)
        )
        self.mlp[2].is_residual = True

    def forward(self, x):
        return self.mlp(x) # (B, T, token_embed)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.token_embed)
        self.attention_heads = CausalSelfAttention(config=config)

        self.ln2 = nn.LayerNorm(config.token_embed)
        self.mlp = FeedForward(config=config)

    def forward(self, x):
        # It's common pratice now to normalize the layers BEFORE the transformation layers,
        # as opposed to AFTER which is what the original Transformer paper introduced.
        residual = self.ln1(x)
        residual = self.attention_heads(residual)

        x = x + residual

        residual = self.ln2(x)
        residual = self.mlp(residual)

        x = x + residual
        return x
    

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            token_embeddings = nn.Embedding(config.num_tokens, config.token_embed), # (num_tokens, token_embed)
            position_embeddings = nn.Embedding(config.context_size, config.token_embed),
            blocks = nn.Sequential(*[Block(config=config) for _ in range(config.num_layers)]),
            ln = nn.LayerNorm(config.token_embed)
        ))

        self.output = nn.Linear(config.token_embed, config.num_tokens, bias=False) # (num_tokens, token_embed)

        # This is the weight tying scheme GPT-2/3 used. We are forcing the
        # model to use a shared vocabulary representation for input and output.
        self.transformer.token_embeddings.weight = self.output.weight

        # Weight initialisation GPT-2/3 used.
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            # Residual parts (the attention output and the second linear in FeedForward)
            # add their output directly into the residual stream. With num_layers blocks, each
            # with 2 residual projections, those additions accumulate and the activations can
            # grow large before training has had a chance to stabilise them via LayerNorm.
            # Scaling down by 1/sqrt(2 * num_layers) keeps the residual stream variance
            # roughly constant from step 0, making early training stable.
            if hasattr(module, 'is_residual'):
                std *= (2 * self.config.num_layers) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, input, targets=None):
        B, T = input.shape

        x = self.transformer.token_embeddings(input) + self.transformer.position_embeddings(torch.arange(T, device=input.device))
        x = self.transformer.blocks(x) # (B, context_size, token_embed)
        x = self.transformer.ln(x) # (B, context_size, token_embed)

        logits = self.output(x) # (B, context_size, num_tokens)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape

            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets, ignore_index=IGNORE_TOKEN)

        return logits, loss

    def generate(self, input, max_new_tokens, context_size):
        for _ in range(max_new_tokens):
            input_trim = input[:, -context_size:]

            logits, _ = self(input_trim)

            # Grab the last token from each example
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            input = torch.cat((input, idx_next), dim=-1)
            
        return input

