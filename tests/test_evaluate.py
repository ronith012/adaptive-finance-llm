import math

import torch
import torch.nn as nn

from afllm.evaluate import evaluate


class _FixedLossModel(nn.Module):
    """Stands in for TransformerLM: returns a prescribed loss per call,
    ignoring the actual input values. Lets us test evaluate()'s
    aggregation logic in isolation, with full control over per-batch loss
    and token count."""

    def __init__(self, losses):
        super().__init__()
        self._losses = iter(losses)
        self._dummy_param = nn.Parameter(torch.zeros(1))  # so .eval()/.train() have something

    def forward(self, x, y=None):
        loss = torch.tensor(next(self._losses), dtype=torch.float32)
        return None, loss


def test_evaluate_known_expected_loss_is_token_weighted_not_batch_averaged():
    # Batch 1: 6 target tokens, loss 2.0. Batch 2: 2 target tokens, loss 8.0.
    # Naive (unweighted) batch-mean would give (2.0 + 8.0) / 2 = 5.0.
    # Correct token-weighted mean: (2.0*6 + 8.0*2) / (6+2) = 28/8 = 3.5.
    # Using unequal token counts deliberately distinguishes the two.
    batch1_x = torch.zeros(1, 6, dtype=torch.long)
    batch1_y = torch.zeros(1, 6, dtype=torch.long)
    batch2_x = torch.zeros(1, 2, dtype=torch.long)
    batch2_y = torch.zeros(1, 2, dtype=torch.long)

    loader = [(batch1_x, batch1_y), (batch2_x, batch2_y)]
    model = _FixedLossModel([2.0, 8.0])

    result = evaluate(model, loader, device="cpu")

    assert result["valid"] is True
    assert result["num_tokens"] == 8
    assert math.isclose(result["loss"], 3.5, rel_tol=1e-9)
    assert result["loss"] != 5.0, "evaluate() is still averaging per-batch, not per-token"
    assert math.isclose(result["perplexity"], math.exp(3.5), rel_tol=1e-9)


def test_evaluate_respects_max_steps():
    # Three batches available, but max_steps=1 should only consume the first.
    loader = [
        (torch.zeros(1, 4, dtype=torch.long), torch.zeros(1, 4, dtype=torch.long)),
        (torch.zeros(1, 4, dtype=torch.long), torch.zeros(1, 4, dtype=torch.long)),
        (torch.zeros(1, 4, dtype=torch.long), torch.zeros(1, 4, dtype=torch.long)),
    ]
    model = _FixedLossModel([1.0, 999.0, 999.0])

    result = evaluate(model, loader, device="cpu", max_steps=1)

    assert result["valid"] is True
    assert result["num_tokens"] == 4
    assert math.isclose(result["loss"], 1.0, rel_tol=1e-9)


def test_evaluate_empty_validation_data_returns_explicit_invalid_not_nan():
    model = _FixedLossModel([])
    result = evaluate(model, loader=[], device="cpu")

    assert result["valid"] is False
    assert result["loss"] is None
    assert result["perplexity"] is None
    assert result["num_tokens"] == 0


def test_evaluate_max_steps_zero_is_also_treated_as_empty():
    model = _FixedLossModel([1.0])
    loader = [(torch.zeros(1, 4, dtype=torch.long), torch.zeros(1, 4, dtype=torch.long))]

    result = evaluate(model, loader, device="cpu", max_steps=0)

    assert result["valid"] is False
    assert result["loss"] is None
    assert result["perplexity"] is None


def test_evaluate_large_loss_reports_finite_perplexity_not_premature_inf():
    # The old threshold (avg_loss < 20) would have reported "inf" here even
    # though math.exp(30) is a perfectly representable finite float.
    model = _FixedLossModel([30.0])
    loader = [(torch.zeros(1, 4, dtype=torch.long), torch.zeros(1, 4, dtype=torch.long))]

    result = evaluate(model, loader, device="cpu")

    assert result["valid"] is True
    assert result["perplexity"] == math.exp(30.0)
    assert result["perplexity"] != float("inf")


def test_evaluate_extreme_loss_still_reports_inf_without_overflowing():
    model = _FixedLossModel([10_000.0])
    loader = [(torch.zeros(1, 4, dtype=torch.long), torch.zeros(1, 4, dtype=torch.long))]

    result = evaluate(model, loader, device="cpu")

    assert result["valid"] is True
    assert result["perplexity"] == float("inf")
