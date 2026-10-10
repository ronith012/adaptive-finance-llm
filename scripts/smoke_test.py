#!/usr/bin/env python
"""Fast, network-free smoke test for the training pipeline.

Verifies model init -> forward -> loss -> backward -> checkpoint
save/reload on a tiny synthetic config, with no dataset download and no
real training. Intended to catch environment/wiring problems (a broken
CUDA/torch install, a bad config, a Drive permissions issue) in seconds,
before committing a Colab session to a full run.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --device cuda
    python scripts/smoke_test.py --checkpoint-dir /content/drive/MyDrive/adaptive-finance-llm/smoke_test
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from afllm.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from afllm.config import ModelConfig  # noqa: E402
from afllm.env_checks import verify_storage_path  # noqa: E402
from afllm.model import TransformerLM  # noqa: E402


def run_smoke_test(device: str, checkpoint_dir: str | None) -> None:
    print(f"[1/5] Building tiny model on device={device} ...")
    cfg = ModelConfig(
        vocab_size=256, dim=32, n_layers=2, n_heads=4, n_kv_heads=2,
        max_seq_len=16, dropout=0.0,
    )
    model = TransformerLM(cfg).to(device)
    print(f"      OK - {model.num_parameters():,} non-embedding parameters")

    print("[2/5] Forward pass + loss computation ...")
    x = torch.randint(0, cfg.vocab_size, (2, 8), device=device)
    y = torch.randint(0, cfg.vocab_size, (2, 8), device=device)
    logits, loss = model(x, y)
    assert logits.shape == (2, 8, cfg.vocab_size), f"unexpected logits shape {logits.shape}"
    assert loss is not None and torch.isfinite(loss), f"loss is not finite: {loss}"
    print(f"      OK - logits shape {tuple(logits.shape)}, loss = {loss.item():.4f}")

    print("[3/5] Backward pass ...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grads_exist = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert grads_exist, "no gradients were computed"
    optimizer.step()
    print("      OK - gradients computed and optimizer step applied")

    cleanup_dir = None
    if checkpoint_dir is None:
        checkpoint_dir = tempfile.mkdtemp(prefix="afllm_smoke_")
        cleanup_dir = checkpoint_dir
        print(f"[4/5] Checkpoint save/reload (local temp dir: {checkpoint_dir}) ...")
    else:
        verify_storage_path(checkpoint_dir)
        print(f"[4/5] Checkpoint save/reload ({checkpoint_dir}) ...")

    path = save_checkpoint(checkpoint_dir, step=0, model=model, optimizer=optimizer)
    model2 = TransformerLM(cfg).to(device)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    state = load_checkpoint(path, model2, optimizer2, map_location=device)
    assert state["step"] == 0
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2.to(p1.device)), "reloaded parameters do not match"
    print(f"      OK - saved to {path} and reloaded with matching parameters")

    if cleanup_dir is not None:
        shutil.rmtree(cleanup_dir, ignore_errors=True)
        print(f"      (cleaned up local temp dir {cleanup_dir})")

    print("[5/5] SMOKE TEST PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on. Defaults to cuda if available, else cpu.",
    )
    parser.add_argument(
        "--checkpoint-dir", default=None,
        help=(
            "Where to write the smoke-test checkpoint. Defaults to a local "
            "temp directory that is deleted afterward. Pass a Drive path "
            "(e.g. /content/drive/MyDrive/adaptive-finance-llm/smoke_test) "
            "to also verify Drive mount + write access; in that case the "
            "checkpoint is left in place rather than deleted."
        ),
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda requested but no GPU is available; falling back to cpu.")
        args.device = "cpu"

    run_smoke_test(args.device, args.checkpoint_dir)


if __name__ == "__main__":
    main()
