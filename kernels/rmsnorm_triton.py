"""
Fused RMSNorm Triton kernel (forward + backward), drop-in for Qwen2RMSNorm.

y = x / sqrt(mean(x^2) + eps) * w

Fusion: one kernel does square-mean + rsqrt + scale (PyTorch eager = 4+ kernels,
4 global-memory round trips). Bandwidth-bound op -> fusion gives real speedup.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd(X, W, Y, RSTD, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(RSTD + row, rstd)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(Y + row * stride + cols, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _rmsnorm_bwd(DY, X, W, RSTD, DX, DW_partial, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD + row)

    xhat = x * rstd
    wdy = w * dy
    # dx = rstd * (wdy - xhat * mean(wdy * xhat))
    c = tl.sum(wdy * xhat, axis=0) / N
    dx = rstd * (wdy - xhat * c)
    tl.store(DX + row * stride + cols, dx.to(DX.dtype.element_ty), mask=mask)
    # per-row dW partials, reduced on host (deterministic, no atomics)
    tl.store(DW_partial + row * N + cols, (dy * xhat), mask=mask)


class TritonRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        shape = x.shape
        x2d = x.reshape(-1, shape[-1]).contiguous()
        M, N = x2d.shape
        y = torch.empty_like(x2d)
        rstd = torch.empty(M, device=x.device, dtype=torch.float32)
        BLOCK = triton.next_power_of_2(N)
        _rmsnorm_fwd[(M,)](x2d, weight, y, rstd, x2d.stride(0), N, eps, BLOCK=BLOCK)
        ctx.save_for_backward(x2d, weight, rstd)
        ctx.shape = shape
        return y.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        x2d, weight, rstd = ctx.saved_tensors
        M, N = x2d.shape
        dy2d = dy.reshape(M, N).contiguous()
        dx = torch.empty_like(x2d)
        dw_partial = torch.empty(M, N, device=x2d.device, dtype=torch.float32)
        BLOCK = triton.next_power_of_2(N)
        _rmsnorm_bwd[(M,)](
            dy2d, x2d, weight, rstd, dx, dw_partial, x2d.stride(0), N, BLOCK=BLOCK
        )
        dw = dw_partial.sum(0).to(weight.dtype)
        return dx.reshape(ctx.shape), dw, None


class TritonRMSNorm(torch.nn.Module):
    """Drop-in replacement for transformers Qwen2RMSNorm."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        return TritonRMSNormFn.apply(x, self.weight, self.variance_epsilon)


def patch_model_rmsnorm(model) -> int:
    """Swap all Qwen2RMSNorm modules for TritonRMSNorm. Returns count."""
    n = 0
    for name, mod in model.named_modules():
        for child_name, child in mod.named_children():
            if child.__class__.__name__ == "Qwen2RMSNorm":
                new = TritonRMSNorm(child.weight.shape[0], child.variance_epsilon)
                new.weight = child.weight
                setattr(mod, child_name, new)
                n += 1
    return n
