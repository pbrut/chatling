import tiktoken


CHAT_SPECIAL_TOKENS = ["<|user_start|>", "<|user_end|>", "<|assistant_start|>", "<|assistant_end|>"]


def build_chat_tokenizer():
    base = tiktoken.get_encoding("gpt2")
    special_tokens = dict(base._special_tokens)
    for i, token in enumerate(CHAT_SPECIAL_TOKENS):
        special_tokens[token] = base.n_vocab + i
    return tiktoken.Encoding(
        name="gpt2_chat",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens=special_tokens,
    )


tokeniser = build_chat_tokenizer()
EOT_TOKEN = tokeniser.encode_single_token("<|endoftext|>")  # reused as conversation-start marker
USER_START_TOKEN = tokeniser.encode_single_token("<|user_start|>")
USER_END_TOKEN = tokeniser.encode_single_token("<|user_end|>")
ASSISTANT_START_TOKEN = tokeniser.encode_single_token("<|assistant_start|>")
ASSISTANT_END_TOKEN = tokeniser.encode_single_token("<|assistant_end|>")

# 50,257 base tokens + 4 custom special tokens = 50,261
# The '<|endoftext|>' special token is already included in the default GPT-2 tokenizer
NUM_TOKENS = tokeniser.n_vocab
