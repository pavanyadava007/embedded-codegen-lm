#!/usr/bin/env python3
"""
Instruction-tuned data pipeline. Extracts complete C functions from the corpus,
forms (instruction, completion) pairs, formats with a fixed prompt template.

Rationale: raw-file continuation (v1) caused catastrophic forgetting of the
base model's task-completion behavior. Pairs align the training objective with
the eval objective (task comment -> implementation).

Usage:
    python data/prepare_instruct.py --out data/processed/embedded_c_instruct
"""
import argparse
import hashlib
import re
import subprocess
from pathlib import Path

from datasets import Dataset

REPOS = [
    ("https://github.com/FreeRTOS/FreeRTOS-Kernel.git", "MIT"),
    ("https://github.com/zephyrproject-rtos/zephyr.git", "Apache-2.0"),
    ("https://github.com/ARM-software/CMSIS_5.git", "Apache-2.0"),
    ("https://github.com/libopencm3/libopencm3.git", "LGPL-3.0"),
    ("https://github.com/eclipse-threadx/threadx.git", "MIT"),
]

# function definition: [static] ret name(params) {   (optional preceding comment)
FUNC_RE = re.compile(
    r"(?P<doc>(?:/\*.*?\*/\s*)|(?:(?:^[ \t]*//[^\n]*\n)+))?"   # optional doc
    r"^(?P<sig>(?:static\s+|inline\s+)*"
    r"[A-Za-z_][\w\s\*]*?\b(?P<name>[A-Za-z_]\w*)\s*"
    r"\([^;{)]*\))\s*"                                          # signature
    r"(?P<body>\{)",                                           # body opener
    re.DOTALL | re.MULTILINE,
)

MIN_BODY, MAX_BODY = 3, 80          # body lines
NAME_BLOCKLIST = re.compile(r"^(if|for|while|switch|return|sizeof|else)$")
TEMPLATE = (
    "// Implement this C99 function (MISRA-compliant, embedded target).\n"
    "{docline}"
    "{sig}\n"
)


def clone(url: str, dst: Path) -> None:
    if not dst.exists():
        subprocess.run(["git", "clone", "--depth", "1", url, str(dst)],
                       check=True, capture_output=True)


def brace_match(text: str, open_idx: int) -> int:
    """Return index just past the matching close brace, or -1."""
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


def clean_doc(doc: str) -> str:
    d = re.sub(r"[/*]", " ", doc)
    d = re.sub(r"\s+", " ", d).strip()
    return d[:200]


def extract(text: str):
    for m in FUNC_RE.finditer(text):
        name = m.group("name")
        if NAME_BLOCKLIST.match(name):         # control keyword, not a function
            continue
        end = brace_match(text, m.start("body"))
        if end == -1:
            continue
        body = text[m.start("body"):end]
        n = body.count("\n")
        if not (MIN_BODY <= n <= MAX_BODY):
            continue
        sig = re.sub(r"\s+", " ", m.group("sig")).strip()
        if len(sig) > 200 or "=" in sig.split("(")[0]:   # reject assignments/initializers
            continue
        doc = clean_doc(m.group("doc") or "")
        docline = f"// {doc}\n" if len(doc) >= 15 else ""
        prompt = TEMPLATE.format(docline=docline, sig=sig)
        completion = " " + body + "\n"
        yield {"prompt": prompt, "completion": completion, "name": name}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/embedded_c_instruct")
    ap.add_argument("--workdir", default="data/raw")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    records, seen = [], set()
    for url, lic in REPOS:
        name = url.rstrip("/").split("/")[-1].removesuffix(".git")
        dst = workdir / name
        clone(url, dst)
        n0 = len(records)
        for p in dst.rglob("*.c"):
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for rec in extract(text):
                h = hashlib.sha256(re.sub(r"\s+", "", rec["completion"]).encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                rec["license"] = lic
                records.append(rec)
        print(f"[{name}] +{len(records) - n0} functions")

    ds = Dataset.from_list(records).train_test_split(test_size=0.02, seed=42)
    ds.save_to_disk(args.out)
    print(f"[done] {len(records)} instruction pairs -> {args.out}")


if __name__ == "__main__":
    main()
