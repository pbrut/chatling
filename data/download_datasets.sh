#!/usr/bin/env bash
set -e

uv run python -m utils.hellaswag
uv run python -m utils.download_sft_data
uv run python -m utils.pre_process_fineweb
