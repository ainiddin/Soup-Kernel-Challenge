"""Correctness of your kernels against plain PyTorch, forward and backward.

Run on a GPU: `python -m pytest solution -q`. Add a test like this for every kernel you write.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-4}


# ── SwiGLU (own fused kernel) ─────────────────────────────────────────────────

@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 8960), (3, 77, 1001)])
def test_swiglu_fused_matches_pytorch(dtype, shape):
    from solution.swiglu_fused import swiglu_fused

    torch.manual_seed(0)
    gate = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    up = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    ref = torch.nn.functional.silu(gate.float()) * up.float()
    ref.backward(grad.float())
    ref_dgate, ref_dup = gate.grad.float(), up.grad.float()
    gate.grad = up.grad = None

    out = swiglu_fused(gate, up)
    out.backward(grad)
    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol)
    torch.testing.assert_close(gate.grad.float(), ref_dgate, rtol=tol, atol=tol)
    torch.testing.assert_close(up.grad.float(), ref_dup, rtol=tol, atol=tol)


# ── SwiGLU (shipped example) ──────────────────────────────────────────────────

@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 8960), (3, 77, 1001)])
def test_swiglu_matches_pytorch(dtype, shape):
    from solution.swiglu import swiglu

    torch.manual_seed(0)
    gate = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    up = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    ref = torch.nn.functional.silu(gate.float()) * up.float()
    ref.backward(grad.float())
    ref_dgate, ref_dup = gate.grad.float(), up.grad.float()
    gate.grad = up.grad = None

    out = swiglu(gate, up)
    out.backward(grad)
    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol)
    torch.testing.assert_close(gate.grad.float(), ref_dgate, rtol=tol, atol=tol)
    torch.testing.assert_close(up.grad.float(), ref_dup, rtol=tol, atol=tol)


# ── RMSNorm ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("shape", [(4, 512, 1536), (2, 64, 512)])
def test_rms_norm_forward_matches_pytorch(shape):
    from solution.rms_norm import fused_rms_norm

    torch.manual_seed(1)
    x = torch.randn(shape, device="cuda", dtype=torch.float16)
    w = torch.ones(shape[-1], device="cuda", dtype=torch.float32)

    # reference: original Qwen2 RMSNorm behaviour
    x_f = x.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    ref = (w * x_f * torch.rsqrt(var + 1e-6)).float()

    out = fused_rms_norm(x, w, 1e-6)
    torch.testing.assert_close(out.float(), ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("shape", [(4, 512, 1536), (2, 64, 512)])
def test_rms_norm_backward_matches_pytorch(shape):
    from solution.rms_norm import fused_rms_norm

    torch.manual_seed(2)
    x_ref = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
    w_ref = torch.ones(shape[-1], device="cuda", dtype=torch.float32, requires_grad=True)
    x_tri = x_ref.detach().clone().requires_grad_(True)
    w_tri = w_ref.detach().clone().requires_grad_(True)
    grad = torch.randn(shape, device="cuda", dtype=torch.float32)

    # Reference forward (matches Qwen2RMSNorm)
    x_f = x_ref.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    ref_out = w_ref * x_f * torch.rsqrt(var + 1e-6)
    ref_out.backward(grad)

    tri_out = fused_rms_norm(x_tri, w_tri, 1e-6)
    tri_out.backward(grad)

    torch.testing.assert_close(x_tri.grad.float(), x_ref.grad.float(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(w_tri.grad.float(), w_ref.grad.float(), rtol=1e-3, atol=1e-3)


# ── Cross-Entropy ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("N,V", [(128, 151936), (64, 32000)])
def test_fused_ce_forward_matches_pytorch(N, V):
    from solution.cross_entropy import fused_cross_entropy

    torch.manual_seed(3)
    logits = torch.randn(N, V, device="cuda", dtype=torch.float32)
    targets = torch.randint(0, V, (N,), device="cuda")
    targets[: N // 8] = -100  # some ignored positions

    ref = torch.nn.functional.cross_entropy(logits, targets, ignore_index=-100)
    out = fused_cross_entropy(logits, targets, ignore_index=-100)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("N,V", [(64, 151936), (32, 32000)])
def test_fused_ce_backward_matches_pytorch(N, V):
    from solution.cross_entropy import fused_cross_entropy

    torch.manual_seed(4)
    logits_ref = torch.randn(N, V, device="cuda", dtype=torch.float32, requires_grad=True)
    logits_tri = logits_ref.detach().clone().requires_grad_(True)
    targets = torch.randint(0, V, (N,), device="cuda")
    targets[: N // 8] = -100

    ref = torch.nn.functional.cross_entropy(logits_ref, targets, ignore_index=-100)
    ref.backward()

    out = fused_cross_entropy(logits_tri, targets, ignore_index=-100)
    out.backward()

    torch.testing.assert_close(logits_tri.grad, logits_ref.grad, rtol=1e-3, atol=1e-3)
