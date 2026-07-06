#!/usr/bin/env bash
# End-to-end pipeline (run on GPU instance).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1. data ===";        python data/prepare_data.py
echo "=== 2. train (LoRA) ==="; python training/train_lora.py --config configs/train_lora.yaml
echo "=== 3. profile ===";      python training/profiling.py --steps 20
echo "=== 4. kernel bench ==="; python kernels/benchmark_kernel.py
echo "=== 5. merge ===";        python serving/merge_lora.py
echo "=== 6. eval: baseline vs fine-tuned ==="
python eval/harness.py --model Qwen/Qwen2.5-Coder-1.5B --k 5 --out eval/results_baseline.json
python eval/harness.py --model checkpoints/merged-embedded-c --k 5 --out eval/results_finetuned.json
echo "=== done ==="
