# embedded-codegen-lm

A code-generation LM specialized for **safety-critical embedded C**, with an end-to-end
pipeline: data → distributed fine-tuning → custom Triton kernel → optimized serving →
**compile / MISRA / unit-test evaluation harness**.

```
embedded C corpora ──► fine-tune (LoRA / FSDP) ──► profile + fused kernel
                                   │
                                   ▼
                     vLLM serving (bf16 / AWQ INT4)
                                   │
                                   ▼
        eval harness: gcc ─► arm-none-eabi-gcc ─► cppcheck --addon=misra ─► unit tests
                                   │
                                   ▼
              compile-rate · MISRA-clean-rate · pass@1 / pass@k
```

## Why this exists

Generic code LMs are evaluated on "does it run". Embedded control software is
evaluated on **compiles for the target, passes MISRA-C, passes tests** - the
harness here encodes those safety gates (informed by ISO 26262 ASIL-B practice)
as automatic metrics, so model quality is measured against the deployment
standard, not a proxy.

## Layout

| Path | Purpose |
|---|---|
| `data/prepare_data.py` | Clone permissive embedded/RTOS repos (FreeRTOS, Zephyr, CMSIS, ThreadX…), filter, dedup, build HF dataset |
| `training/train_lora.py` | Single-GPU LoRA fine-tune of Qwen2.5-Coder-1.5B (bf16, grad-ckpt, packing) |
| `training/train_fsdp.py` | Multi-GPU FSDP full fine-tune; logs tokens/s and **MFU** |
| `training/profiling.py` | `torch.profiler` Chrome trace: data / fwd / bwd / optim breakdown |
| `kernels/rmsnorm_triton.py` | Fused RMSNorm Triton kernel (fwd + bwd, autograd, model patcher) |
| `kernels/benchmark_kernel.py` | Correctness check + latency/bandwidth vs PyTorch eager |
| `serving/` | LoRA merge → vLLM OpenAI-compatible server → AWQ INT4 quant → async throughput/latency benchmark |
| `eval/harness.py` | Generate → compile → cross-compile (Cortex-M4) → MISRA → unit test; unbiased pass@k |
| `eval/selftest.py` | Verifies the harness itself on hand-written MISRA-clean reference solutions |

## Quickstart (GPU instance)

```bash
pip install -r requirements.txt
apt-get install -y cppcheck gcc-arm-none-eabi     # MISRA + cross-compile gates
bash scripts/run_all.sh                            # full pipeline
```

Individual steps:

```bash
python data/prepare_data.py
python training/train_lora.py --config configs/train_lora.yaml
torchrun --nproc_per_node=2 training/train_fsdp.py          # Tier 2: FSDP
python kernels/benchmark_kernel.py                           # Tier 2: kernel
python serving/merge_lora.py && bash serving/serve_vllm.sh   # Tier 3: serving
python serving/benchmark_serving.py --n 64 --concurrency 8
python eval/harness.py --model checkpoints/merged-embedded-c --k 5
```

Verify harness without a GPU:

```bash
python eval/selftest.py
# [PASS] ring_buffer  compile=True tests=True misra=clean
# [PASS] crc16        compile=True tests=True misra=clean
# [PASS] pid          compile=True tests=True misra=clean
# [PASS] debounce     compile=True tests=True misra=clean
# [PASS] moving_avg   compile=True tests=True misra=clean
```

## Results (measured on one NVIDIA L4, July and September 2026)

15 tasks x 5 samples per task, same harness (`eval/harness.py`) for every model. Raw per-task numbers: `eval/results_baseline.json`, `eval/results_finetuned_15.json`, `eval/results_instruct.json`.

| Metric | Base Qwen2.5-Coder-1.5B | LoRA v1: raw-file continuation | LoRA v2: instruction pairs |
|---|---|---|---|
| compile rate (host gcc) | 0.73 | 0.47 | 0.80 |
| cross-compile rate (arm-none-eabi-gcc, Cortex-M4) | 0.60 | 0.45 | 0.80 |
| MISRA-clean rate (cppcheck addon) | 0.19 | 0.27 | 0.16 |
| pass@1 (unbiased estimator, n=5) | 0.16 | 0.12 | 0.63 |
| pass@5 | 0.60 | 0.33 | 0.87 |

Per-task pass@1:

| Task | Base | LoRA v1 | LoRA v2 |
|---|---|---|---|
| ring_buffer | 0.00 | 0.00 | 1.00 |
| crc16 | 0.00 | 0.00 | 0.00 |
| pid | 0.40 | 0.20 | 0.40 |
| debounce | 0.20 | 0.00 | 1.00 |
| moving_avg | 0.00 | 0.00 | 0.00 |
| clamp_i32 | 0.40 | 0.00 | 1.00 |
| map_range | 0.00 | 0.40 | 0.40 |
| parity | 0.20 | 0.00 | 0.40 |
| reverse_bits | 0.20 | 0.40 | 0.80 |
| sat_add_u8 | 0.40 | 0.20 | 1.00 |
| lerp_fixed | 0.00 | 0.00 | 0.20 |
| count_leading_zeros | 0.00 | 0.00 | 0.40 |
| gcd | 0.20 | 0.00 | 0.80 |
| is_power_of_two | 0.20 | 0.00 | 1.00 |
| array_max | 0.20 | 0.60 | 1.00 |

What the numbers say:

- **LoRA v1 made the model worse**: pass@1 0.16 to 0.12 and compile rate 0.73 to 0.47. Training on raw file continuation (57k functions from FreeRTOS, Zephyr, CMSIS, libopencm3 and ThreadX) taught the model to continue files, not to answer a task comment. `data/prepare_instruct.py` documents the fix.
- **LoRA v2 (instruction pairs: doc comment + signature -> body) lifted pass@1 from 0.16 to 0.63**, pass@5 from 0.60 to 0.87 and cross-compile rate from 0.60 to 0.80.
- **MISRA-clean rate did not improve** (0.19 base, 0.16 v2). The training corpora are not MISRA-clean, so tuning on them does not teach compliance. The next step would be filtering the corpus through the MISRA gate before training.
- Scale caveat: 15 tasks and 5 samples is a demo-size benchmark, and sampling makes single runs noisy; differences of one or two tasks are within noise.
- History: the first v1 evaluation (July) covered only the first 5 tasks; it was re-run on all 15 in September 2026 so that all three columns use the same task set.
- Contamination check: training function names were compared with the eval signatures; `crc16_ccitt` exists in the Zephyr corpus (the eval asks for CRC-16/CCITT-FALSE). All other eval function names are absent from the training set.

Not run yet, so no numbers are reported for them: the FSDP multi-GPU run, the fused RMSNorm Triton benchmark and the vLLM bf16 vs AWQ INT4 serving benchmark.

## Design decisions (interview-ready)

- **LoRA before full FT** - domain adaptation of a 1.5B model saturates with r=16
  adapters at a fraction of the compute; FSDP path exists to demonstrate full-shard
  training and MFU accounting, not because it's required for quality.
- **Sequence packing** - embedded C files are short; packing to 2048 removes
  padding waste (~35-60% token utilization gain typical on this corpus).
- **Fused RMSNorm** - RMSNorm is bandwidth-bound; eager = 4+ kernel launches and
  4 HBM round trips, fusion = 1. Backward implemented without atomics
  (per-row dW partials + host reduction) → deterministic gradients.
- **MISRA gate at eval, not decode** - constrained decoding against MISRA is
  intractable (rules are semantic); post-hoc gating + fine-tuning on compliant
  corpora moves the distribution instead.
- **Unbiased pass@k** (Chen et al. 2021 estimator), not naive best-of-k, so
  numbers are comparable to HumanEval-style reporting.
- **MISRA suppressions** - rules 2.x (unused-entity) and 8.4/8.7/8.9 (linkage/
  declaration placement) are artifacts of single-snippet evaluation, not code
  quality; suppressed and documented.

## Known limits

- cppcheck's MISRA addon covers a subset of MISRA-C:2012 (rule texts not bundled);
  a commercial checker (PC-lint, Helix QAC) would be used in production.
- 5 eval tasks = demo scale; extend `eval/prompts/tasks.jsonl` + matching tests.
- Cross-compile gate checks compilation only; running on-target (QEMU Cortex-M or
  HIL) is the natural next step.
