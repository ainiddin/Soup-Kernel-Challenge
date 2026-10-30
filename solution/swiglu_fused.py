"""Own fused SwiGLU for the Qwen2 MLP: silu(gate) * up, forward and backward.

This is my own implementation, independent of the shipped example:
- the forward keeps the sigmoid and the silu value as explicit intermediates
  and reuses them, rather than folding everything into one expression;
- the backward uses the decomposition silu'(x) = s + silu*(1 - s)
  (with s = sigmoid(x), silu = x*s) instead of s*(1 + x*(1 - s));
- a configurable 1-D block over the flattened activation.

silu(x) = x * sigmoid(x)
d/dgate [silu(gate) * up] = up * silu'(gate),  silu'(x) = s + silu*(1 - s)
d/dup   [silu(gate) * up] = silu(gate)
"""

import torch
import triton
import triton.language as tl

TILE = 2048


@triton.jit
def swiglu_fwd(gate_ptr, up_ptr, out_ptr, n, TILE: tl.constexpr):
    pid = tl.program_id(axis=0)
    idx = pid * TILE + tl.arange(0, TILE)
    keep = idx < n
    g = tl.load(gate_ptr + idx, mask=keep, other=0.0).to(tl.float32)
    u = tl.load(up_ptr + idx, mask=keep, other=0.0).to(tl.float32)
    s = tl.sigmoid(g)
    silu = g * s
    result = silu * u
    tl.store(out_ptr + idx, result.to(out_ptr.dtype.element_ty), mask=keep)


@triton.jit
def swiglu_bwd(dout_ptr, gate_ptr, up_ptr, dgate_ptr, dup_ptr, n, TILE: tl.constexpr):
    pid = tl.program_id(axis=0)
    idx = pid * TILE + tl.arange(0, TILE)
    keep = idx < n
    dout = tl.load(dout_ptr + idx, mask=keep, other=0.0).to(tl.float32)
    g = tl.load(gate_ptr + idx, mask=keep, other=0.0).to(tl.float32)
    u = tl.load(up_ptr + idx, mask=keep, other=0.0).to(tl.float32)
    s = tl.sigmoid(g)
    silu = g * s
    # d(up)   = dout * silu(gate)
    grad_up = dout * silu
    # d(gate) = dout * up * silu'(gate),  silu'(x) = s + silu*(1 - s)
    dsilu = s + silu * (1.0 - s)
    grad_gate = dout * u * dsilu
    tl.store(dgate_ptr + idx, grad_gate.to(dgate_ptr.dtype.element_ty), mask=keep)
    tl.store(dup_ptr + idx, grad_up.to(dup_ptr.dtype.element_ty), mask=keep)


class _SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        gate = gate.contiguous()
        up = up.to(gate.dtype).contiguous()
        out = torch.empty_like(gate)
        n = gate.numel()
        grid = (triton.cdiv(n, TILE),)
        swiglu_fwd[grid](gate, up, out, n, TILE=TILE)
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, dout):
        gate, up = ctx.saved_tensors
        dout = dout.to(gate.dtype).contiguous()
        dgate = torch.empty_like(gate)
        dup = torch.empty_like(up)
        n = gate.numel()
        grid = (triton.cdiv(n, TILE),)
        swiglu_bwd[grid](dout, gate, up, dgate, dup, n, TILE=TILE)
        return dgate, dup


def swiglu_fused(gate, up):
    return _SwiGLU.apply(gate, up)


def patch_qwen2_mlp_own():
    """Route Qwen2MLP.forward through my own fused SwiGLU kernel."""
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2MLP.forward

    def forward(self, x):
        act = getattr(getattr(self, "config", None), "hidden_act", "silu")
        if act != "silu" or not x.is_cuda:
            return original(self, x)
        return self.down_proj(swiglu_fused(self.gate_proj(x), self.up_proj(x)))

    modeling_qwen2.Qwen2MLP.forward = forward
