#!/usr/bin/env python3
"""
Serving benchmark against the vLLM endpoint:
reports throughput (tok/s), TTFT, and p50/p95/p99 latency.

Usage:
    python serving/benchmark_serving.py --n 64 --concurrency 8
"""
import argparse
import asyncio
import json
import statistics
import time

import aiohttp

PROMPT = ("// Implement CRC-16/CCITT-FALSE in MISRA-compliant C99.\n"
          "#include <stdint.h>\n")


async def one_request(session, url, max_tokens):
    t0 = time.perf_counter()
    ttft = None
    n_tok = 0
    payload = {"model": "served", "prompt": PROMPT,
               "max_tokens": max_tokens, "stream": True, "temperature": 0.6}
    async with session.post(url, json=payload) as resp:
        async for line in resp.content:
            if not line.startswith(b"data: "):
                continue
            if line.strip() == b"data: [DONE]":
                break
            if ttft is None:
                ttft = time.perf_counter() - t0
            chunk = json.loads(line[6:])
            n_tok += len(chunk["choices"][0].get("text", "")) > 0
    return time.perf_counter() - t0, ttft or 0.0, n_tok


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1/completions")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=256)
    args = ap.parse_args()

    sem = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession() as s:
        async def bounded():
            async with sem:
                return await one_request(s, args.url, args.max_tokens)
        t0 = time.perf_counter()
        results = await asyncio.gather(*[bounded() for _ in range(args.n)])
        wall = time.perf_counter() - t0

    lats = sorted(r[0] for r in results)
    ttfts = [r[1] for r in results]
    total_tok = sum(r[2] for r in results)
    q = lambda p: lats[min(int(p * len(lats)), len(lats) - 1)]
    print(f"requests            : {args.n} @ concurrency {args.concurrency}")
    print(f"throughput          : {total_tok / wall:,.0f} tok/s")
    print(f"TTFT mean           : {statistics.mean(ttfts)*1e3:.0f} ms")
    print(f"latency p50/p95/p99 : {q(.5):.2f} / {q(.95):.2f} / {q(.99):.2f} s")


if __name__ == "__main__":
    asyncio.run(main())
