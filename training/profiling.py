#!/usr/bin/env python3
"""
torch.profiler trace of the training step: data loading vs forward vs backward vs optim.
Exports Chrome trace (open in chrome://tracing or Perfetto).

Usage:
    python training/profiling.py --config configs/train_lora.yaml --steps 20
"""
import argparse

import torch
import yaml
from datasets import load_from_disk
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_lora import pack_sequences


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_lora.yaml")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", default="profiling/trace")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=torch.bfloat16
    ).to(device)
    model.gradient_checkpointing_enable()

    ds = load_from_disk(cfg["dataset_path"])["train"].select(range(2000))
    ds = pack_sequences(ds, tokenizer, cfg["max_seq_len"])
    ds.set_format("torch")
    loader = DataLoader(ds, batch_size=cfg["per_device_batch_size"], num_workers=4)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    sched = schedule(wait=2, warmup=3, active=args.steps - 5, repeat=1)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(args.out),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        it = iter(loader)
        for _ in range(args.steps):
            batch = {k: v.to(device) for k, v in next(it).items()}
            out = model(**batch)
            out.loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
            prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
    print(f"[done] trace in {args.out}/ (view with tensorboard --logdir {args.out})")


if __name__ == "__main__":
    main()
