#!/usr/bin/env python
"""CLI entrypoint for a one-time, final evaluation on the held-out test split.

Deliberately separate from scripts/train.py and never imported by
src/afllm/train.py: this script is meant to be run once, manually, after
training has finished and a model has been selected - never during
training, and never used to make training or hyperparameter decisions.

Usage:
    python scripts/evaluate_test.py --config configs/baseline_small.yaml
    python scripts/evaluate_test.py --config configs/baseline_small.yaml \
        --checkpoint /content/drive/MyDrive/adaptive-finance-llm/checkpoints/ckpt_step499.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from afllm.checkpoint import find_latest_checkpoint, load_checkpoint  # noqa: E402
from afllm.config import load_config, validate_vocab_size  # noqa: E402
from afllm.data import load_wikitext2, make_test_loader  # noqa: E402
from afllm.env_checks import resolve_device, verify_training_paths  # noqa: E402
from afllm.evaluate import evaluate  # noqa: E402
from afllm.model import TransformerLM  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML experiment config")
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to a specific checkpoint file. Defaults to the latest "
        "checkpoint in the config's train.checkpoint_dir.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Limit evaluation to this many batches. Defaults to the full test split.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    verify_training_paths(cfg)

    # Resolve the checkpoint (and fail fast if there isn't one) before
    # touching the dataset at all - no point tokenizing the full test split
    # only to then discover there's nothing to evaluate.
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint(
        cfg.train.checkpoint_dir
    )
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"No checkpoint found in {cfg.train.checkpoint_dir!r} and no "
            f"--checkpoint was given."
        )

    device = resolve_device(cfg.train.device)

    corpus = load_wikitext2(cfg.data)
    validate_vocab_size(cfg, corpus.vocab_size)
    test_loader = make_test_loader(corpus, cfg.data, cfg.train.batch_size)

    model = TransformerLM(cfg.model).to(device)
    # require_compatible_config=cfg: unlike train()'s resume path (which
    # only logs a mismatch and proceeds), a final test-set evaluation
    # refuses outright on a confirmed critical-field mismatch, raising
    # CheckpointConfigMismatchError with the specific fields that differ -
    # before any weights are applied to `model` - rather than silently
    # evaluating a model that was never actually produced by this config.
    state = load_checkpoint(
        checkpoint_path, model, optimizer=None, restore_rng=False,
        map_location=device, require_compatible_config=cfg,
    )

    result = evaluate(model, test_loader, device, max_steps=args.max_steps)
    result["checkpoint"] = str(checkpoint_path)
    result["checkpoint_step"] = state["step"]
    print(json.dumps(result))


if __name__ == "__main__":
    main()
