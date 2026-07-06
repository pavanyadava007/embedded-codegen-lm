#!/usr/bin/env python3
"""
Benchmark: TritonRMSNorm vs PyTorch eager RMSNorm (Qwen2 reference impl).
Checks numerical correctness, then reports latency + effective bandwidth.

Usage:
    python kernels/benchmark_kernel.py
"""
import torch
import triton

from rmsnorm_triton import TritonRMSNorm


class EagerRMSNorm(torch.nn.Module):
    """Reference: transformers Qwen2RMSNorm (eager)."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x.to(dt)).to(dt)


def bench(fn, x):
    ms = triton.testing.do_bench(lambda: fn(x), warmup=25, rep=100)
    gb = 2 * x.numel() * x.element_size() / 1e9  # read + write
    return ms, gb / (ms / 1e3)


def main():
    torch.manual_seed(0)
    device, dtype = "cuda", torch.bfloat16
    N = 1536  # Qwen2.5-1.5B hidden size

    print(f"{'shape':<20}{'eager ms':>10}{'triton ms':>11}{'speedup':>9}{'GB/s':>8}")
    for M in [512, 2048, 8192, 32768]:
        x = torch.randn(M, N, device=device, dtype=dtype, requires_grad=True)
        eager, fused = EagerRMSNorm(N).to(device, dtype), TritonRMSNorm(N).to(device, dtype)

        # correctness (fwd + bwd)
        y_ref, y_tr = eager(x), fused(x)
        torch.testing.assert_close(y_ref, y_tr, rtol=1e-2, atol=1e-2)
        g = torch.randn_like(y_ref)
        (gx_ref,) = torch.autograd.grad(y_ref, x, g, retain_graph=True)
        (gx_tr,) = torch.autograd.grad(y_tr, x, g, retain_graph=True)
        torch.testing.assert_close(gx_ref, gx_tr, rtol=1e-2, atol=1e-2)

        ms_e, _ = bench(eager, x.detach())
        ms_t, bw = bench(fused, x.detach())
        print(f"({M}, {N})".ljust(20)
              + f"{ms_e:>10.4f}{ms_t:>11.4f}{ms_e/ms_t:>8.2f}x{bw:>8.0f}")

    print("\ncorrectness: PASS (fwd + bwd, rtol=1e-2)")


if __name__ == "__main__":
    main()
