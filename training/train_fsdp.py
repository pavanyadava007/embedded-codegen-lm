#!/usr/bin/env python3
"""
FSDP full fine-tune across N GPUs (Tier 2). Logs throughput + MFU.

Usage (2 GPUs):
    torchrun --nproc_per_node=2 training/train_fsdp.py --config configs/train_lora.yaml
"""
import argparse
import functools
import math
import os
import time

import torch
import torch.distributed as dist
import yaml
from datasets import load_from_disk
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from train_lora import pack_sequences  # reuse packing


def mfu(tokens_per_s: float, n_params: float, peak_flops: float) -> float:
    """Model FLOPs Utilization: achieved / peak. ~6*N FLOPs per token (fwd+bwd)."""
    return (6 * n_params * tokens_per_s) / peak_flops


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_lora.yaml")
    ap.add_argument("--peak_tflops", type=float, default=125.0,
                    help="per-GPU peak BF16 TFLOPs (A10G=125, A100=312)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    torch.manual_seed(cfg["seed"] + rank)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=torch.bfloat16
    )
    n_params = sum(p.numel() for p in model.parameters())

    wrap_policy = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls={Qwen2DecoderLayer}
    )
    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=local_rank,
        use_orig_params=True,
    )
    model.gradient_checkpointing_enable()

    ds = load_from_disk(cfg["dataset_path"])["train"]
    ds = pack_sequences(ds, tokenizer, cfg["max_seq_len"])
    ds.set_format("torch")
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
    loader = DataLoader(
        ds, batch_size=cfg["per_device_batch_size"], sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    optim = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    accum = cfg["gradient_accumulation_steps"]

    model.train()
    step, tokens_local, t0 = 0, 0, time.time()
    for epoch in range(cfg["num_epochs"]):
        sampler.set_epoch(epoch)
        for i, batch in enumerate(loader):
            batch = {k: v.to(local_rank, non_blocking=True) for k, v in batch.items()}
            out = model(**batch)
            (out.loss / accum).backward()
            tokens_local += batch["input_ids"].numel()

            if (i + 1) % accum == 0:
                model.clip_grad_norm_(1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                if step % cfg["logging_steps"] == 0 and rank == 0:
                    dt = time.time() - t0
                    tps_global = tokens_local * world / dt
                    u = mfu(tps_global / world, n_params, args.peak_tflops * 1e12)
                    print(
                        f"step {step} loss {out.loss.item():.4f} "
                        f"tok/s(global) {tps_global:,.0f} MFU {u*100:.1f}%"
                    )

    if rank == 0:
        # gather full state dict on rank 0
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        cfg_sd = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg_sd):
            sd = model.state_dict()
        torch.save(sd, f"{cfg['output_dir']}/fsdp_final.pt")
        print("[done] saved fsdp_final.pt")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
