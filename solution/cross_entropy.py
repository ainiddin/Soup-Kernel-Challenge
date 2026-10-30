"""Fused cross-entropy in Triton.

Strategy: patch Qwen2ForCausalLM.forward to strip `labels` from the call,
get logits from the original forward, then compute CE ourselves with Triton
kernels. This bypasses all loss_function / CrossEntropyLoss binding issues
since we never touch the model's internal loss path.
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

    # Pass 1: row max for numerical stability
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
    # Ignored rows (label == ignore_index) do not contribute to the loss, so
    # their gradient must be exactly zero. Zero the scale for those rows.
    scale = tl.where(label == ignore_index, 0.0, upstream / valid_count)

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
    """Patch Qwen2ForCausalLM.forward: strip labels, get logits, compute CE in Triton.

    Works regardless of how the model internally binds CrossEntropyLoss or
    loss_function — we bypass the model's loss path entirely.
    """
    from transformers.models.qwen2 import modeling_qwen2

    _orig = modeling_qwen2.Qwen2ForCausalLM.forward

    def _patched(self, *args, **kwargs):
        labels = kwargs.pop("labels", None)

        # labels may also come positionally (index 5 in Qwen2ForCausalLM.forward)
        # signature: input_ids, attention_mask, position_ids, past_key_values,
        #            inputs_embeds, labels, ...
        pos_labels = None
        if labels is None and len(args) > 5:
            lst = list(args)
            pos_labels = lst[5]
            lst[5] = None
            args = tuple(lst)

        effective_labels = labels if labels is not None else pos_labels

        # Call original forward without labels so it returns logits only
        out = _orig(self, *args, **kwargs)

        if effective_labels is None:
            return out

        logits = out.logits  # (B, T, V)
        if logits is None:
            return out

        # Standard causal-LM shift: predict token t+1 from position t
        shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        shift_labels = effective_labels[..., 1:].contiguous().view(-1).to(shift_logits.device)

        loss = fused_cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        out.loss = loss
        return out

    modeling_qwen2.Qwen2ForCausalLM.forward = _patched
