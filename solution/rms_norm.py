"""Fused RMSNorm forward and backward for Qwen2 in Triton.

Replaces the original multi-kernel PyTorch path (cast→pow→mean→rsqrt→scale→cast)
with two single-pass Triton kernels (one forward, one backward), reducing memory
traffic and kernel launch overhead for all 57 RMSNorm calls per training step.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_fwd_kernel(
    X, W, Y, Rstd,
    stride, N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
    tl.store(Y + row * stride + cols, x * rstd * w, mask=mask)


@triton.jit
def rms_norm_bwd_kernel(
    X, W, DY, DX, Rstd,
    stride, N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
    rstd = tl.load(Rstd + row).to(tl.float32)
    dyw = dy * w
    c1 = tl.sum(dyw * x, axis=0) * (rstd * rstd) / N
    dx = rstd * (dyw - x * c1)
    tl.store(DX + row * stride + cols, dx.to(tl.float16), mask=mask)


class _FusedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        M = x2.shape[0]
        BLOCK = triton.next_power_of_2(N)
        y = torch.empty(M, N, device=x.device, dtype=torch.float32)
        rstd = torch.empty(M, device=x.device, dtype=torch.float32)
        rms_norm_fwd_kernel[(M,)](x2, weight, y, rstd, N, N, eps, BLOCK=BLOCK)
        ctx.save_for_backward(x2, weight, rstd)
        ctx.BLOCK = BLOCK
        ctx.N = N
        return y.view(orig_shape)

    @staticmethod
    def backward(ctx, dy):
        x2, weight, rstd = ctx.saved_tensors
        M, N = x2.shape
        dy2 = dy.contiguous().view(M, N).to(torch.float32)
        dx = torch.empty(M, N, device=x2.device, dtype=torch.float16)
        rms_norm_bwd_kernel[(M,)](x2, weight, dy2, dx, rstd, N, N, BLOCK=ctx.BLOCK)
        x_normed = x2.float() * rstd[:, None]
        dw = (dy2 * x_normed).sum(0).to(weight.dtype)
        return dx.view_as(dy).to(dy.dtype), dw, None


def fused_rms_norm(x, weight, eps):
    return _FusedRMSNorm.apply(x, weight, eps)


_original_qwen2_rms_forward = None


def patch_qwen2_rms_norm():
    global _original_qwen2_rms_forward
    from transformers.models.qwen2 import modeling_qwen2

    _original_qwen2_rms_forward = modeling_qwen2.Qwen2RMSNorm.forward

    def forward(self, hidden_states):
        if not hidden_states.is_cuda:
            return _original_qwen2_rms_forward(self, hidden_states)
        return fused_rms_norm(hidden_states, self.weight, self.variance_epsilon)

    modeling_qwen2.Qwen2RMSNorm.forward = forward
