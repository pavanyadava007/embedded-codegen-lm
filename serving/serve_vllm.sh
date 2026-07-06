#!/usr/bin/env bash
# Serve merged model with vLLM (OpenAI-compatible API).
# Merge LoRA first:  python serving/merge_lora.py
set -euo pipefail
MODEL="${1:-checkpoints/merged-embedded-c}"
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --port 8000
