"""Fused cross-entropy in Triton.

Qwen2ForCausalLM does `from torch.nn import CrossEntropyLoss` at module level,
so patching nn.CrossEntropyLoss.forward is too late — the local name is already
bound. We replace modeling_qwen2.CrossEntropyLoss with a subclass whose forward
calls our Triton kernels for the LM-head shape (N, 151936).
"""

import torch
import triton
import triton.language as tl

BLOCK_V = 1024


@triton.jit
def fused_ce_fwd_kernel(
    logits_ptr, labels_ptr, loss_ptr, lse_ptr,
    N, V, ignore_index,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    label = tl.load(labels_ptr + row)

    # Pass 1: row maximum for numerical stability
    m = -1e9
    for v in range(0, V, BLOCK_V):
        cols = v + tl.arange(0, BLOCK_V)
        mask = cols < V
        x = tl.load(logits_ptr + row * V + cols, mask=mask, other=-1e9).to(tl.float32)
        m = tl.maximum(m, tl.max(x, axis=0))

    # Pass 2: log-sum-exp
    s = 0.0
    for v in range(0, V, BLOCK_V):
        cols = v + tl.arange(0, BLOCK_V)
        mask = cols < V
        x = tl.load(logits_ptr + row * V + cols, mask=mask, other=-1e9).to(tl.float32)
        s += tl.sum(tl.where(mask, tl.exp(x - m), 0.0), axis=0)

    lse = m + tl.log(s)
    tl.store(lse_ptr + row, lse)

    if label != ignore_index:
        tgt = tl.load(logits_ptr + row * V + label).to(tl.float32)
        tl.store(loss_ptr + row, lse - tgt)
    else:
        tl.store(loss_ptr + row, 0.0)


@triton.jit
def fused_ce_bwd_kernel(
    logits_ptr, labels_ptr, dlogits_ptr, upstream_ptr, lse_ptr,
    N, V, valid_count, ignore_index,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    label = tl.load(labels_ptr + row)
    lse = tl.load(lse_ptr + row).to(tl.float32)
    upstream = tl.load(upstream_ptr).to(tl.float32)
    scale = upstream / valid_count

    for v in range(0, V, BLOCK_V):
        cols = v + tl.arange(0, BLOCK_V)
        mask = cols < V
        x = tl.load(logits_ptr + row * V + cols, mask=mask, other=-1e9).to(tl.float32)
        softmax = tl.where(mask, tl.exp(x - lse), 0.0)
        is_target = mask & (cols == label) & (label != ignore_index)
        dlogit = (softmax - tl.where(is_target, 1.0, 0.0)) * scale
        tl.store(dlogits_ptr + row * V + cols, dlogit, mask=mask)


class _FusedCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, ignore_index):
        N, V = logits.shape
        logits_f = logits.float().contiguous()
        labels_c = labels.contiguous()
        loss_per_row = torch.empty(N, device=logits.device, dtype=torch.float32)
        lse = torch.empty(N, device=logits.device, dtype=torch.float32)
        fused_ce_fwd_kernel[(N,)](
            logits_f, labels_c, loss_per_row, lse,
            N, V, ignore_index, BLOCK_V=BLOCK_V,
        )
        valid = (labels_c != ignore_index).sum().clamp(min=1)
        loss = loss_per_row.sum() / valid
        ctx.save_for_backward(logits_f, labels_c, lse)
        ctx.ignore_index = ignore_index
        ctx.valid_count = int(valid.item())
        return loss.to(logits.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        logits_f, labels_c, lse = ctx.saved_tensors
        N, V = logits_f.shape
        dlogits = torch.empty_like(logits_f)
        upstream = grad_output.float().contiguous()
        fused_ce_bwd_kernel[(N,)](
            logits_f, labels_c, dlogits, upstream, lse,
            N, V, ctx.valid_count, ctx.ignore_index,
            BLOCK_V=BLOCK_V,
        )
        return dlogits.to(grad_output.dtype), None, None


def fused_cross_entropy(logits, labels, ignore_index=-100):
    return _FusedCE.apply(logits, labels, ignore_index)


def patch_cross_entropy():
    """Replace modeling_qwen2.CrossEntropyLoss with a fused subclass.

    Qwen2ForCausalLM imports CrossEntropyLoss by name at module load time:
        from torch.nn import CrossEntropyLoss
    Patching nn.CrossEntropyLoss after that has no effect on that local name.
    We must replace the name in the module's namespace directly.
    """
    from transformers.models.qwen2 import modeling_qwen2
    import torch.nn as nn

    _orig = getattr(modeling_qwen2, "CrossEntropyLoss", nn.CrossEntropyLoss)

    class _FusedCELoss(_orig):
        def forward(self, input, target):
            if (input.dim() == 2
                    and input.is_cuda
                    and input.shape[-1] > 32000):
                return fused_cross_entropy(input, target,
                                           ignore_index=self.ignore_index)
            return super().forward(input, target)

    modeling_qwen2.CrossEntropyLoss = _FusedCELoss

    # Also patch nn.CrossEntropyLoss.forward as a belt-and-suspenders fallback
    # (covers any Soup/TRL code that instantiates CrossEntropyLoss after apply())
    _orig_nn_fwd = nn.CrossEntropyLoss.forward

    def _nn_fwd(self, input, target):
        if (input.dim() == 2
                and input.is_cuda
                and input.shape[-1] > 32000):
            return fused_cross_entropy(input, target,
                                       ignore_index=self.ignore_index)
        return _orig_nn_fwd(self, input, target)

    nn.CrossEntropyLoss.forward = _nn_fwd
