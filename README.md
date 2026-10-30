# Soup Kernel Challenge — Our Solution

Custom Triton kernels that make `soup train` (Qwen2.5-1.5B-Instruct, 4-bit NF4, LoRA r=16,
64 steps) **lighter on GPU memory** on a free Kaggle T4, without changing what it trains.

## Results on Kaggle T4

All 10 checks of `harness/check_rules.py` pass.

| | Base Soup 0.77.0 | Ours | Ratio |
|---|---|---|---|
| Time of `soup train` | baseline | ≈ same | speedup **x1.00** (tie) |
| Peak GPU memory | baseline | **−17.9%** | base/ours **x1.217** |
| Eval loss (public held-out rows) | baseline | identical | **−0.004%** vs base |
| Our kernels' share of training time | — | **9.7%** | (check 4 needs ≥ 5%) |

- Loss curve follows the base: first step within 0.02%, mean of all steps within 0.26%.
- Soup's outputs (steps, adapter, trainer settings, tokens, metrics, rewind) identical within
  tolerance.
- Public Kaggle notebook: **<paste your public notebook link here>**

## What we did

We wrote three pairs of fused forward/backward Triton kernels and patched Soup's model so the
training run uses them. The headline win is **memory**: the fused SwiGLU and the chunked
cross-entropy avoid materialising large intermediate tensors.

### Our kernels (`solution/`)

| Kernel (`@triton.jit`) | File | Replaces | Why it helps |
|---|---|---|---|
| `swiglu_fwd`, `swiglu_bwd` | `solution/swiglu_fused.py` | `Qwen2MLP` SiLU-gate path (3 PyTorch ops) | one fused launch each way; no materialised `silu(gate)` and `silu(gate)*up` intermediates → ~2 GB less peak memory |
| `fused_ce_fwd_kernel`, `fused_ce_bwd_kernel` | `solution/cross_entropy.py` | LM-head cross-entropy over the 151,936-token vocabulary | online (two-pass) softmax in the forward; the backward writes the gradient in-place over vocab tiles → never stores a full `(N, 151936)` softmax matrix |
| `rms_norm_fwd_kernel`, `rms_norm_bwd_kernel` | `solution/rms_norm.py` | all 57 `Qwen2RMSNorm` calls per step | single-pass normalisation (cast→pow→mean→rsqrt→scale fused); the backward caches `rstd` from the forward |

- **SwiGLU**: `silu(x) = x·sigmoid(x)`, implemented as `x/(1+exp(-x))`; backward uses
  `silu'(x) = s + silu·(1−s)`. A grid-stride loop over `STEPS` tiles per program.
- **Cross-entropy**: one CTA per row loops over the vocab in `BLOCK_V` chunks. Forward is a
  stable two-pass (row max, then log-sum-exp); backward recomputes softmax per tile and writes
  `(softmax − one_hot(label)) · scale`, with the gradient forced to **zero for ignored (`-100`)
  rows** so the masked prompt tokens contribute nothing — this is what keeps the loss curve on
  the base.
- **RMSNorm**: forward stores `rstd` per row; backward reuses it and reduces `dy·w·x` in one pass.

Each kernel has a correctness test against plain PyTorch in `solution/test_kernels.py`
(`python -m pytest solution -q` on a CUDA GPU).

### How the kernels plug in

`solution/apply()` patches, at call time:
- `Qwen2MLP.forward` → our fused SwiGLU;
- `Qwen2RMSNorm.forward` → our fused RMSNorm;
- `Qwen2ForCausalLM.forward` → strips `labels`, runs the original forward for logits, then
  computes the loss with our cross-entropy kernels (bypasses the internal loss path entirely,
  which is the only version-robust interception point in transformers 5.16.1).

All patches are guarded for non-CUDA / non-`nn.Module` inputs so Soup's regression tests
(`tests/`, check 10) still pass.

## How to run it

On Kaggle (**GPU T4 x2**, **Internet on**):
1. Import `templates/kaggle_notebook.ipynb`.
2. Set the repo URL to this repository and `COMMIT` to the final commit SHA (see `SUBMISSION.md`).
3. **Run All** (~10 min). You get the base and our solution side by side, `runs/result.json`,
   and the table of checks.

Locally on a Linux machine with a CUDA T4:

```bash
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
    "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python task/prepare_data.py
python -m pytest solution -q            # our kernels vs plain PyTorch
python harness/run_task.py              # base, then solution -> runs/result.json
python harness/check_rules.py           # the 10 checks -> runs/checks.json
```

Only numbers from a Kaggle T4 count (Soup trains in fp16 there; Ampere+ uses a different bf16 path).

## Limits and next steps

- **Speed is a tie, not a win.** The savings are in memory; on the T4 the fused kernels roughly
  match the base in wall time (x1.00). The step time is dominated by the 4-bit matmuls and NF4
  dequantization, which we did not touch.
- The cross-entropy forward still casts logits to fp32 and the backward writes a full `(N, V)`
  gradient tensor (fused in-place, but still allocated). A logit-chunked backward that keeps only
  `d_hidden` would cut memory further.
- We kept the shipped example SwiGLU in the repo but do **not** use it; our `swiglu_fused.py`
  runs instead.
- Next: fuse the LoRA `q_proj`/`v_proj` matmuls, and a Triton NF4 dequant to attack the ~5% that
  dequantization costs.

## License

Apache-2.0, see `LICENSE` and `NOTICE`. Test files from Soup v0.77.0. Data:
[AmanMussa/kazakh-instruction-v2](https://huggingface.co/datasets/AmanMussa/kazakh-instruction-v2).
Model: [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).
