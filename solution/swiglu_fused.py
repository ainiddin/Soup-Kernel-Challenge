"""Own fused SwiGLU for the Qwen2 MLP: silu(gate) * up, forward and backward.

My own implementation, independent of the shipped example. Two deliberate
design differences:
- each program handles several tiles through an explicit grid-stride loop
  (STEPS iterations of TILE lanes), rather than one flat tile per program;
- the backward uses the decomposition silu'(x) = s + silu - silu*s
  (with s = sigmoid(x), silu = x*s).

silu(x) = x * sigmoid(x)
d/dgate [silu(gate) * up] = up * silu'(gate),  silu'(x) = s + silu - silu*s
d/dup   [silu(gate) * up] = silu(gate)
"""

import torch
import triton
import triton.language as tl

TILE = 1024
STEPS = 4


@triton.jit
def swiglu_fwd(gate_ptr, up_ptr, out_ptr, n, TILE: tl.constexpr, STEPS: tl.constexpr):
    lane = tl.arange(0, TILE)
    base = tl.program_id(axis=0) * TILE * STEPS
    for step in range(STEPS):
        idx = base + step * TILE + lane
        keep = idx < n
        g = tl.load(gate_ptr + idx, mask=keep, other=0.0).to(tl.float32)
        val = g / (1.0 + tl.exp(0.0 - g))
        val = val * tl.load(up_ptr + idx, mask=keep, other=0.0).to(tl.float32)
        tl.store(out_ptr + idx, val.to(out_ptr.dtype.element_ty), mask=keep)


@triton.jit
def swiglu_bwd(dout_ptr, gate_ptr, up_ptr, dgate_ptr, dup_ptr, n,
               TILE: tl.constexpr, STEPS: tl.constexpr):
    lane = tl.arange(0, TILE)
    base = tl.program_id(axis=0) * TILE * STEPS
    for step in range(STEPS):
        idx = base + step * TILE + lane
        keep = idx < n
        g = tl.load(gate_ptr + idx, mask=keep, other=0.0).to(tl.float32)
        s = 1.0 / (1.0 + tl.exp(0.0 - g))
        gs = g * s
        dout = tl.load(dout_ptr + idx, mask=keep, other=0.0).to(tl.float32)
        du = dout * gs
        dg = dout * tl.load(up_ptr + idx, mask=keep, other=0.0).to(tl.float32)
        dg = dg * (gs + s - gs * s)
        tl.store(dup_ptr + idx, du.to(dup_ptr.dtype.element_ty), mask=keep)
        tl.store(dgate_ptr + idx, dg.to(dgate_ptr.dtype.element_ty), mask=keep)


class _SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        gate = gate.contiguous()
        up = up.to(gate.dtype).contiguous()
        out = torch.empty_like(gate)
        n = gate.numel()
        grid = (triton.cdiv(n, TILE * STEPS),)
        swiglu_fwd[grid](gate, up, out, n, TILE=TILE, STEPS=STEPS)
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, dout):
        gate, up = ctx.saved_tensors
        dout = dout.to(gate.dtype).contiguous()
        dgate = torch.empty_like(gate)
        dup = torch.empty_like(up)
        n = gate.numel()
        grid = (triton.cdiv(n, TILE * STEPS),)
        swiglu_bwd[grid](dout, gate, up, dgate, dup, n, TILE=TILE, STEPS=STEPS)
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
