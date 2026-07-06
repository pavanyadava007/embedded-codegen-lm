#!/usr/bin/env python3
"""AWQ INT4 quantization of the merged model (smaller + faster serving)."""
import argparse
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="checkpoints/merged-embedded-c")
ap.add_argument("--out", default="checkpoints/merged-embedded-c-awq")
args = ap.parse_args()

model = AutoAWQForCausalLM.from_pretrained(args.model)
tok = AutoTokenizer.from_pretrained(args.model)
model.quantize(tok, quant_config={
    "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"
})
model.save_quantized(args.out)
tok.save_pretrained(args.out)
print(f"[done] AWQ INT4 -> {args.out}")
