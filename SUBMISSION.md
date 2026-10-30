# Submission: <team name>

## Links
- Repository, final commit: https://github.com/ainiddin/Soup-Kernel-Challenge/commit/30ae7aaec65cc2b04c2acd0761b21bb166cfae17
  (also in the team repo: https://github.com/BAITC-Hacks/hack-0f0c42c6-world)
- Public Kaggle notebook (saved version pinned to `COMMIT = 30ae7aa...`; same link is in README):
  https://www.kaggle.com/code/ainiddin/notebook2697177e12
- Demo video (up to 3 minutes): <link>
- Presentation: <link>

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| | | ainiddin | | |

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `swiglu_fwd`, `swiglu_bwd` | `solution/swiglu_fused.py` | Qwen2MLP SiLU-gate (`silu(gate)*up`) | part of 9.7% |
| `fused_ce_fwd_kernel`, `fused_ce_bwd_kernel` | `solution/cross_entropy.py` | LM-head cross-entropy over 151,936 vocab | part of 9.7% |
| `rms_norm_fwd_kernel`, `rms_norm_bwd_kernel` | `solution/rms_norm.py` | all 57 `Qwen2RMSNorm` calls per step | part of 9.7% |

Total own-kernel share: **9.7%** of the training time (check 4, ≥ 5% required).

Ideas taken from other kernels, with links (we wrote the code ourselves):
- Standard online-softmax / log-sum-exp cross-entropy idea (two-pass, per-row), general public
  knowledge; code is our own.

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json`)
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | baseline | ≈ same | speedup x1.004 (tie, counts as 1.000) |
| Peak GPU memory, MiB | baseline | −17.9% | base/ours x1.217 |
| Eval loss (public rows) | baseline | identical | −0.004% vs base |

- [x] All 10 checks of `harness/check_rules.py` passed
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Which operations we sped up and why (SwiGLU, cross-entropy, RMSNorm — chosen because element-wise
ops, dtype casts and the big-vocab softmax dominate the memory traffic), how each fused kernel
works, results against the base (memory −17.9%, speed tie, quality identical), and limits.

## Limitations
- Memory win, speed tie: the step is bottlenecked by the 4-bit matmuls and NF4 dequantization,
  which we did not fuse. The cross-entropy backward still allocates a full `(N, V)` gradient.
- Next: Triton NF4 dequant and fused LoRA matmuls to turn the memory win into a time win.

---

**Final commit used for the measured run:** `c881c78a3e737e3bb81f4615f369f4b521fe3565`
Set this exact SHA as `COMMIT` in the saved Kaggle notebook version.
