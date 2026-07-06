#!/usr/bin/env python3
"""
LoRA fine-tune Qwen2.5-Coder on embedded C corpus. Single-GPU (Tier 1).

Usage:
    python training/train_lora.py --config configs/train_lora.yaml
"""
import argparse
import math
import time

import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)


def pack_sequences(dataset, tokenizer, max_len: int):
    """Concatenate + chunk into fixed-length blocks (standard causal-LM packing)."""
    def tokenize_fn(batch):
        return tokenizer(batch["text"], add_special_tokens=False)

    tokenized = dataset.map(
        tokenize_fn, batched=True, remove_columns=dataset.column_names
    )

    def group_fn(batch):
        ids = []
        for x in batch["input_ids"]:
            ids.extend(x + [tokenizer.eos_token_id])
        n = (len(ids) // max_len) * max_len
        chunks = [ids[i : i + max_len] for i in range(0, n, max_len)]
        return {"input_ids": chunks, "labels": [c[:] for c in chunks]}

    return tokenized.map(group_fn, batched=True, remove_columns=tokenized.column_names)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_lora.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    torch.manual_seed(cfg["seed"])
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch.bfloat16 if cfg["bf16"] else torch.float32,
        attn_implementation="sdpa",
    ).to(device)

    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_from_disk(cfg["dataset_path"])["train"]
    ds = pack_sequences(ds, tokenizer, cfg["max_seq_len"])
    ds.set_format("torch")
    loader = DataLoader(
        ds, batch_size=cfg["per_device_batch_size"], shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    accum = cfg["gradient_accumulation_steps"]
    steps_per_epoch = math.ceil(len(loader) / accum)
    total_steps = steps_per_epoch * cfg["num_epochs"]
    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=cfg["learning_rate"]
    )
    sched = get_cosine_schedule_with_warmup(
        optim, int(cfg["warmup_ratio"] * total_steps), total_steps
    )

    model.train()
    step, t0, tokens_seen = 0, time.time(), 0
    for epoch in range(cfg["num_epochs"]):
        for i, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            out = model(**batch)
            (out.loss / accum).backward()
            tokens_seen += batch["input_ids"].numel()

            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                if step % cfg["logging_steps"] == 0:
                    dt = time.time() - t0
                    print(
                        f"epoch {epoch} step {step}/{total_steps} "
                        f"loss {out.loss.item():.4f} "
                        f"tok/s {tokens_seen / dt:,.0f} "
                        f"lr {sched.get_last_lr()[0]:.2e}"
                    )
                if step % cfg["save_steps"] == 0:
                    model.save_pretrained(f"{cfg['output_dir']}/step-{step}")

    model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"[done] saved to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
