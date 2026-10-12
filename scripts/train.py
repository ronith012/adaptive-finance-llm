#!/usr/bin/env python
"""CLI entrypoint for training. Intended to run in Colab with Drive mounted.

Usage:
    python scripts/train.py --config configs/baseline_small.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from afllm.config import load_config  # noqa: E402
from afllm.data import load_wikitext2, make_loaders  # noqa: E402
from afllm.env_checks import verify_training_paths  # noqa: E402
from afllm.train import train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML experiment config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    verify_training_paths(cfg)

    corpus = load_wikitext2(cfg.data)
    train_loader, val_loader = make_loaders(corpus, cfg.data, cfg.train.batch_size)

    def log_fn(entry: dict) -> None:
        print(json.dumps(entry))

    train(cfg, train_loader, val_loader, tokenizer_vocab_size=corpus.vocab_size, log_fn=log_fn)


if __name__ == "__main__":
    main()
