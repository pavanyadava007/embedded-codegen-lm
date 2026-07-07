#!/usr/bin/env python3
"""
Instruction LoRA fine-tune with completion-only loss masking.
Prompt tokens set to label -100 -> loss computed only on the implementation.

Fixes v1 degradation: objective now matches eval (generate body given task+sig),
and the model is not penalized for reproducing the given prompt.

Usage:
    python training/train_instruct.py --config configs/train_instruct.yaml
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

IGNORE = -100


def build_examples(ds, tokenizer, max_len):
    def fmt(ex):
        p_ids = tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(ex["completion"], add_special_tokens=False)["input_ids"]
        c_ids = c_ids + [tokenizer.eos_token_id]
        ids = (p_ids + c_ids)[:max_len]
        labels = ([IGNORE] * len(p_ids) + c_ids)[:max_len]
        return {"input_ids": ids, "labels": labels}

    return ds.map(fmt, remove_columns=ds.column_names)


def collate(batch, pad_id):
    m = max(len(x["input_ids"]) for x in batch)
    ids, lbl, att = [], [], []
    for x in batch:
        n = m - len(x["input_ids"])
        ids.append(x["input_ids"] + [pad_id] * n)
        lbl.append(x["labels"] + [IGNORE] * n)
        att.append([1] * len(x["input_ids"]) + [0] * n)
    return {
        "input_ids": torch.tensor(ids),
        "labels": torch.tensor(lbl),
        "attention_mask": torch.tensor(att),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_instruct.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    torch.manual_seed(cfg["seed"])
    device = "cuda"

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model = get_peft_model(model, LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"], task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    ds = load_from_disk(cfg["dataset_path"])["train"]
    ds = build_examples(ds, tok, cfg["max_seq_len"])
    loader = DataLoader(
        ds, batch_size=cfg["per_device_batch_size"], shuffle=True,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
        num_workers=4, pin_memory=True, drop_last=True,
    )

    accum = cfg["gradient_accumulation_steps"]
    total = math.ceil(len(loader) / accum) * cfg["num_epochs"]
    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=cfg["learning_rate"]
    )
    sched = get_cosine_schedule_with_warmup(
        optim, int(cfg["warmup_ratio"] * total), total
    )

    model.train()
    step, t0, seen = 0, time.time(), 0
    for epoch in range(cfg["num_epochs"]):
        for i, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            loss = model(**batch).loss
            (loss / accum).backward()
            seen += int(batch["attention_mask"].sum())
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
                step += 1
                if step % cfg["logging_steps"] == 0:
                    print(f"epoch {epoch} step {step}/{total} loss {loss.item():.4f} "
                          f"tok/s {seen/(time.time()-t0):,.0f} lr {sched.get_last_lr()[0]:.2e}",
                          flush=True)
                if step % cfg["save_steps"] == 0:
                    model.save_pretrained(f"{cfg['output_dir']}/step-{step}")

    model.save_pretrained(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])
    print(f"[done] saved to {cfg['output_dir']}", flush=True)


if __name__ == "__main__":
    main()
