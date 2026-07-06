#!/usr/bin/env python3
"""
Harness self-test: runs compile/MISRA/test gates on reference solutions
(no model, no GPU). All 5 tasks must pass -> harness verified.

Usage:
    python eval/selftest.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness import evaluate_sample  # noqa: E402

ROOT = Path(__file__).resolve().parent
TASKS = {json.loads(l)["task_id"]: json.loads(l)
         for l in open(ROOT / "prompts/tasks.jsonl")}
REF = {
    "ring_buffer": "ring_buffer.c",
    "crc16": "crc16.c",
    "pid": "pid.c",
    "debounce": "debounce.c",
    "moving_avg": "moving_avg.c",
}


def main() -> int:
    failed = 0
    for task_id, ref_file in REF.items():
        t = TASKS[task_id]
        completion = (ROOT / "reference" / ref_file).read_text()
        r = evaluate_sample(t["prompt"], completion, ROOT / "tests" / t["test_file"])
        ok = r["compiles"] and r["tests_pass"]
        status = "PASS" if ok else "FAIL"
        misra = "clean" if r["misra_clean"] else f"violations={r['misra_violations']}"
        print(f"[{status}] {task_id:<12} compile={r['compiles']} "
              f"tests={r['tests_pass']} misra={misra}")
        failed += (not ok)
    print("\nharness selftest:", "PASS" if failed == 0 else f"FAIL ({failed})")
    return failed


if __name__ == "__main__":
    sys.exit(main())
