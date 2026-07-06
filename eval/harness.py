#!/usr/bin/env python3
"""
Evaluation harness for embedded-C code generation.

Pipeline per sample:
  1. GENERATE   model completes the task prompt (k samples, temperature sampling)
  2. COMPILE    host gcc -std=c99 -Wall -Werror  (functional gate)
  3. CROSS      arm-none-eabi-gcc -mcpu=cortex-m4 (embedded-target gate, optional)
  4. MISRA      cppcheck --addon=misra           (safety gate)
  5. TEST       run unit tests                    (correctness gate)

Metrics: compile-rate, cross-compile-rate, MISRA-clean-rate,
         pass@1 / pass@k (Chen et al. 2021 unbiased estimator).

Usage:
    python eval/harness.py --model checkpoints/lora-embedded-c --k 5
    python eval/harness.py --model Qwen/Qwen2.5-Coder-1.5B --k 5   # baseline
"""
import argparse
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GCC = shutil.which("gcc")
ARM_GCC = shutil.which("arm-none-eabi-gcc")
CPPCHECK = shutil.which("cppcheck")

# MISRA rules ignored at eval time (advisory / not meaningful for snippets)
MISRA_SUPPRESS = {"2.3", "2.4", "2.5", "8.4", "8.7", "8.9"}
# 8.4 suppressed: snippet eval has no separate header declaration by design.


def extract_code(text: str) -> str:
    """Take generated completion; strip markdown fences if present."""
    m = re.search(r"```(?:c)?\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


def run(cmd, timeout=30, cwd=None):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, cwd=cwd)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"


def misra_violations(source: Path) -> list[str]:
    """cppcheck MISRA addon; returns rule IDs violated (after suppressions)."""
    if CPPCHECK is None:
        return []
    _, out = run([CPPCHECK, "--addon=misra", "--enable=style",
                  "--inline-suppr", str(source)], timeout=60)
    rules = re.findall(r"misra-c2012-([\d.]+)", out)
    return sorted({r for r in rules if r not in MISRA_SUPPRESS})


def evaluate_sample(prompt: str, completion: str, test_file: Path) -> dict:
    res = {"compiles": False, "cross_compiles": False,
           "misra_clean": False, "misra_violations": [], "tests_pass": False}
    code = prompt + extract_code(completion)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        sol_c = td / "solution.c"
        sol_h = td / "solution.h"
        sol_c.write_text(code)
        # header = the full code guarded (simple approach for snippet eval)
        sol_h.write_text(
            "#ifndef SOLUTION_H\n#define SOLUTION_H\n" + code + "\n#endif\n"
        )

        # 1. host compile (as translation unit)
        rc, _ = run([GCC, "-std=c99", "-Wall", "-Werror", "-c",
                     str(sol_c), "-o", str(td / "sol.o")])
        res["compiles"] = rc == 0
        if not res["compiles"]:
            return res

        # 2. cross-compile for Cortex-M4 (if toolchain present)
        if ARM_GCC:
            rc, _ = run([ARM_GCC, "-mcpu=cortex-m4", "-mthumb", "-std=c99",
                         "-Wall", "-Werror", "-c", str(sol_c),
                         "-o", str(td / "sol_arm.o")])
            res["cross_compiles"] = rc == 0
        else:
            res["cross_compiles"] = None  # toolchain absent

        # 3. MISRA
        v = misra_violations(sol_c)
        res["misra_violations"] = v
        res["misra_clean"] = len(v) == 0

        # 4. unit tests (header-only inclusion; test provides main)
        test_bin = td / "test"
        rc, out = run([GCC, "-std=c99", "-I", str(td),
                       str(test_file), "-o", str(test_bin), "-lm"])
        if rc == 0:
            rc, out = run([str(test_bin)], timeout=10)
            res["tests_pass"] = (rc == 0) and ("PASS" in out)
    return res


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k (Chen et al., 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def generate_completions(model_path: str, prompt: str, k: int,
                         max_new: int = 512) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not hasattr(generate_completions, "_cache"):
        tok = AutoTokenizer.from_pretrained(model_path)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="cuda"
        )
        generate_completions._cache = (tok, mdl)
    tok, mdl = generate_completions._cache
    inp = tok(prompt, return_tensors="pt").to("cuda")
    out = mdl.generate(
        **inp, max_new_tokens=max_new, do_sample=True, temperature=0.6,
        top_p=0.95, num_return_sequences=k,
        pad_token_id=tok.eos_token_id,
    )
    return [tok.decode(o[inp["input_ids"].shape[1]:], skip_special_tokens=True)
            for o in out]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tasks", default=str(ROOT / "prompts/tasks.jsonl"))
    ap.add_argument("--out", default="eval/results.json")
    args = ap.parse_args()

    tasks = [json.loads(l) for l in open(args.tasks)]
    all_results, agg = [], {"compile": 0, "cross": 0, "misra": 0}
    pass1_sum, passk_sum, total_samples = 0.0, 0.0, 0

    for t in tasks:
        completions = generate_completions(args.model, t["prompt"], args.k)
        sample_results = [
            evaluate_sample(t["prompt"], c, ROOT / "tests" / t["test_file"])
            for c in completions
        ]
        n = len(sample_results)
        c_pass = sum(r["tests_pass"] for r in sample_results)
        agg["compile"] += sum(r["compiles"] for r in sample_results)
        agg["cross"] += sum(bool(r["cross_compiles"]) for r in sample_results)
        agg["misra"] += sum(r["misra_clean"] for r in sample_results)
        total_samples += n
        p1, pk = pass_at_k(n, c_pass, 1), pass_at_k(n, c_pass, args.k)
        pass1_sum += p1
        passk_sum += pk
        all_results.append({"task_id": t["task_id"], "pass@1": p1,
                            f"pass@{args.k}": pk, "samples": sample_results})
        print(f"[{t['task_id']}] pass@1={p1:.2f} pass@{args.k}={pk:.2f} "
              f"compile={sum(r['compiles'] for r in sample_results)}/{n}")

    summary = {
        "model": args.model,
        "k": args.k,
        "compile_rate": agg["compile"] / total_samples,
        "cross_compile_rate": agg["cross"] / total_samples,
        "misra_clean_rate": agg["misra"] / total_samples,
        "pass@1": pass1_sum / len(tasks),
        f"pass@{args.k}": passk_sum / len(tasks),
    }
    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    json.dump({"summary": summary, "tasks": all_results},
              open(args.out, "w"), indent=2)
    print("\n=== SUMMARY ===")
    for k_, v in summary.items():
        print(f"{k_:>20}: {v if isinstance(v, str) else round(v, 3)}")


if __name__ == "__main__":
    main()
