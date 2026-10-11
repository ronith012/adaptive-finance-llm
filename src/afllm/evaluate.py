"""Validation loop: average loss and perplexity over a held-out set.

Loss is aggregated token-weighted (sum of per-batch loss*token_count,
divided by total token count), not as a plain mean-of-batch-means. With
`TokenBlockDataset` + `drop_last=True`, every batch currently has the same
number of target tokens, so the two approaches happen to agree today - but
token-weighting is correct regardless of that invariant, and stays correct
if batching ever changes (variable-length batches, padding with
`ignore_index`, a different dataset).
"""

from __future__ import annotations

import math

import torch

# math.exp overflows above ~709.78 for a float64 argument; stay well clear
# of that boundary rather than reporting "inf" for merely-large-but-finite
# perplexities (the previous threshold of 20 was overly conservative -
# exp(20) is nowhere near overflow).
_MAX_EXP_ARG = 700.0


def _safe_perplexity(avg_loss: float) -> float:
    if avg_loss > _MAX_EXP_ARG:
        return float("inf")
    try:
        return math.exp(avg_loss)
    except OverflowError:
        return float("inf")


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: str, max_steps: int | None = None) -> dict:
    """Returns a dict with keys:
      "valid": bool            - False if no batches were available (e.g.
          an empty validation set) or max_steps == 0; when False, "loss"
          and "perplexity" are None, never NaN.
      "loss": float | None     - token-weighted mean cross-entropy loss.
      "perplexity": float | None
      "num_tokens": int        - total target tokens the loss was computed
          over, for anyone who wants to double-check the weighting.

    Callers must check "valid" before logging "loss"/"perplexity" as
    numbers - an invalid result must not silently flow into logs as NaN.
    """
    model.eval()
    total_loss_sum = 0.0
    total_tokens = 0
    for i, (x, y) in enumerate(loader):
        if max_steps is not None and i >= max_steps:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        # `loss` is cross_entropy's mean over non-ignored target tokens in
        # this batch; multiplying back by that same count and summing
        # across batches, then dividing by the grand total, reproduces the
        # true token-weighted mean regardless of whether batches are
        # uniformly sized.
        num_tokens = int((y != -100).sum().item())
        total_loss_sum += loss.item() * num_tokens
        total_tokens += num_tokens
    model.train()

    if total_tokens == 0:
        return {"valid": False, "loss": None, "perplexity": None, "num_tokens": 0}

    avg_loss = total_loss_sum / total_tokens
    return {
        "valid": True,
        "loss": avg_loss,
        "perplexity": _safe_perplexity(avg_loss),
        "num_tokens": total_tokens,
    }
