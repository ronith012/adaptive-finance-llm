"""Validation loop: average loss and perplexity over a held-out set."""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: str, max_steps: int | None = None) -> dict:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for i, (x, y) in enumerate(loader):
        if max_steps is not None and i >= max_steps:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        n_batches += 1
    model.train()

    if n_batches == 0:
        return {"loss": float("nan"), "perplexity": float("nan")}

    avg_loss = total_loss / n_batches
    return {"loss": avg_loss, "perplexity": math.exp(avg_loss) if avg_loss < 20 else float("inf")}
