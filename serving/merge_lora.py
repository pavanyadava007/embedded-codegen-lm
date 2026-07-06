#!/usr/bin/env python3
"""Merge LoRA adapter into base model for vLLM serving."""
import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-1.5B")
ap.add_argument("--adapter", default="checkpoints/lora-embedded-c")
ap.add_argument("--out", default="checkpoints/merged-embedded-c")
args = ap.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, args.adapter)
model = model.merge_and_unload()
model.save_pretrained(args.out)
AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
print(f"[done] merged -> {args.out}")
