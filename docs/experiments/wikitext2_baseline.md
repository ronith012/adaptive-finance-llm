# WikiText-2 baseline smoke test

> **Status: PRELIMINARY, AND FROM A DIFFERENT IMPLEMENTATION.** The numbers
> below are carried over from the project owner's original Colab baseline
> (learned positional embeddings, LayerNorm, standard GELU MLP — see
> [`docs/architecture_notes.md`](../architecture_notes.md)). **They have not
> been reproduced with this repository's RoPE + RMSNorm + SwiGLU
> implementation** (`src/afllm/model.py`). No training run has been executed
> against this repo's code as of this writing. Do not cite these numbers as
> validating, benchmarking, or characterizing this repository's model —
> they document a prior, separate experiment only, kept here for historical
> reference.

## Setup (as reported for the original run)

- Dataset: WikiText-2 (raw), tokenized with the GPT-2 tokenizer
- Hardware: Google Colab, Tesla T4
- Steps: 500
- Model: the project owner's original baseline architecture (learned
  positional embeddings, standard causal self-attention, LayerNorm, GELU
  MLP, tied embeddings) — **not** `src/afllm/model.py` in this repository.
- No hyperparameter search performed; defaults only.
- Exact original code is not present in this repository, so this run
  cannot currently be independently reproduced from this repo's history.

## Results (as reported, unverified by this repository's code)

| Metric | Value |
|---|---|
| Training loss (step 500) | ≈ 5.5916 |
| Validation loss (step 500) | ≈ 5.6771 |

Both figures are in a plausible range for cross-entropy loss over a ~50k
token vocabulary this early in training (loss near `ln(vocab_size) ≈ 10.8`
at initialization, decreasing as the model fits). Train and validation loss
being close together is expected at this stage — 500 steps is far too few to
see overfitting emerge. This plausibility check is **not** the same as
verification; the run itself was not reproduced or inspected log-by-log as
part of this repository.

## Before this repo's implementation gets its own entry here

1. Run `scripts/train.py` with [`configs/baseline_small.yaml`](../../configs/baseline_small.yaml)
   (RoPE + RMSNorm + SwiGLU) end-to-end on Colab, and log the result as a
   **new, separate** entry — do not overwrite or merge it with the numbers
   above, since they come from a different architecture.
2. Retain `train.log_dir` JSONL logs and attach/link them for
   reproducibility.
3. Run for substantially more steps than 500 before drawing any trend
   conclusion.
4. Compare against a trivial baseline (e.g. unigram frequency model) and
   report perplexity, not just loss, before claiming the model learned
   something non-trivial.
5. Run more than once (different seeds) before treating any single number
   as representative.
6. If a true apples-to-apples comparison against the original baseline is
   ever wanted, that requires porting the original architecture into this
   repo (or the new one into the original notebook) and running both under
   matched seed/steps/hardware — not comparing numbers from two different
   models as if they were the same experiment.
