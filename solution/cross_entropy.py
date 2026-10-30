"""Fused cross-entropy in Triton.

In transformers 5.16.1, Qwen2ForCausalLM sets self.loss_function = ForCausalLMLoss
(a plain function) in __init__. We replace that attribute directly on the model
instance via the _setup_transformers hook — the most reliable interception point.
Belt-and-suspenders: also patch F.cross_entropy and any direct reference in
transformers.loss.loss_utils for other callers.
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

    # Pass 2: sum-exp → lse
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


def _make_fused_loss_function(ignore_index=-100):
    """Drop-in replacement for transformers ForCausalLMLoss."""
    def _loss(logits, labels, vocab_size=None, num_items_in_batch=None,
              ignore_index=ignore_index, **kwargs):
        shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        shift_labels = labels[..., 1:].contiguous().view(-1).to(shift_logits.device)
        return fused_cross_entropy(shift_logits, shift_labels, ignore_index=ignore_index)
    return _loss


def patch_cross_entropy():
    import torch.nn.functional as F

    # ── Path 1: patch F.cross_entropy (covers any code that looks it up on the module) ──
    _orig_f = F.cross_entropy

    def _patched_f(input, target, weight=None, size_average=None, ignore_index=-100,
                   reduce=None, reduction="mean", label_smoothing=0.0):
        if (input.dim() == 2 and input.is_cuda
                and label_smoothing == 0.0 and reduction == "mean"
                and weight is None and input.shape[-1] > 32000):
            return fused_cross_entropy(input, target, ignore_index=ignore_index)
        return _orig_f(input, target, weight=weight, size_average=size_average,
                       ignore_index=ignore_index, reduce=reduce, reduction=reduction,
                       label_smoothing=label_smoothing)

    F.cross_entropy = _patched_f
    torch.nn.functional.cross_entropy = _patched_f

    # ── Path 2: replace the function reference inside transformers.loss.loss_utils ──
    try:
        import transformers.loss.loss_utils as _lu
        import torch.nn as _nn
        # The module uses nn.functional — redirect it
        _lu.nn.functional.cross_entropy = _patched_f  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        pass

    # ── Path 3: hook _setup_transformers to replace model.loss_function directly ──
    # This is the most reliable path: after setup the model instance exists and
    # we can swap out its loss_function attribute in-place.
    try:
        from soup_cli.trainer.sft import SFTTrainerWrapper

        _orig_setup = SFTTrainerWrapper._setup_transformers

        def _patched_setup(self, *args, **kwargs):
            result = _orig_setup(self, *args, **kwargs)
            if not isinstance(getattr(self, "model", None), torch.nn.Module):
                return result
            # Unwrap PEFT layers to reach the base CausalLM
            inner = self.model
            for attr in ("base_model", "model"):
                while hasattr(inner, attr):
                    inner = getattr(inner, attr)
                    if hasattr(inner, "loss_function"):
                        break
            if hasattr(inner, "loss_function"):
                cfg_ign = getattr(getattr(inner, "config", None), "ignore_index", -100)
                inner.loss_function = _make_fused_loss_function(ignore_index=cfg_ign)
            return result

        SFTTrainerWrapper._setup_transformers = _patched_setup
    except (ImportError, AttributeError):
        pass
