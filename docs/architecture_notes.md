# Architecture notes: this implementation vs. the original Colab baseline

> **This implementation is a provisional research baseline, not a port of
> the original Colab baseline.** The two are architecturally different
> models. Do not treat results from one as evidence about the other.

## What changed, and why it matters

| Component | Original Colab baseline | This implementation ([src/afllm/model.py](../src/afllm/model.py)) | Equivalent? |
|---|---|---|---|
| Position information | Learned positional embeddings (a trainable per-position vector added to the token embedding) | Rotary position embeddings (RoPE) — query/key vectors are rotated as a function of relative position; no position embedding table exists | **No.** Different parameterization, different inductive bias, different parameter count. |
| Attention | Standard causal self-attention | Causal self-attention via `scaled_dot_product_attention`, implemented to also support grouped-query attention (fewer KV heads than query heads); the shipped config (`baseline_small.yaml`) sets `n_kv_heads == n_heads`, so it computes the same thing as standard MHA today | Numerically similar *as currently configured*, but the code is not "standard attention" — it's a superset that happens to be configured down to standard behavior. |
| Normalization | LayerNorm (mean-subtract + rescale + learned bias) | RMSNorm (rescale by root-mean-square only, no mean-subtraction, no bias term) | **No.** Different statistics, fewer parameters, different gradient behavior. |
| MLP | Standard GELU MLP (two linear layers, GELU in between) | SwiGLU (three linear layers, SiLU-gated) | **No.** Gated activation, different parameter count for the same hidden width. |
| Residual connections | Present | Present (pre-norm, around attention and MLP) | Yes. |
| Tied embeddings | Present | Present, config-gated (`tie_embeddings: true` by default) | Yes. |

## Why this isn't "equivalent, just modernized"

RoPE, RMSNorm, and SwiGLU are individually well-studied substitutions, but
each changes the function the model computes and how it trains — different
initialization sensitivity, different training dynamics, different
parameter budget at a given `dim`/`n_layers`. Swapping all three
simultaneously means any comparison against the original baseline's
numbers conflates at least three independent variables. There is currently
**no controlled comparison** between the two architectures in this repo —
no shared seed, no shared step count run on both, no shared hardware run
logged side by side.

## Status of this implementation

Treat `src/afllm/model.py` as a standalone, provisional research baseline:
useful for iterating on efficient-architecture experiments going forward,
but **not a verified reproduction** of your original Colab results. See
[`docs/experiments/wikitext2_baseline.md`](experiments/wikitext2_baseline.md)
for the one existing WikiText-2 data point, which predates this
implementation and has not been reproduced with it.

## Scope of this note

No additional architectural changes were made alongside this
documentation — this file only records what already existed in
`model.py` and corrects the record on how it relates to the original
baseline. Any future architecture change (e.g. porting the original
LayerNorm/learned-position-embedding/GELU-MLP design into this repo for a
true side-by-side comparison) should be a separate, explicitly-scoped
change with its own experiment log.
