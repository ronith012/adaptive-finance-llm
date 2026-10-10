"""Checkpoint save/load and recovery.

Checkpoints are plain `torch.save` dicts containing model, optimizer,
scheduler, RNG, and step state, written to `TrainConfig.checkpoint_dir`
(a Google Drive path in the Colab workflow). `find_latest_checkpoint`
lets training resume after a Colab disconnect without re-running
from step 0.

Reproducibility scope: saving/restoring Python's `random`, NumPy, and
PyTorch (CPU + CUDA) RNG states means that, from the restored step onward,
anything that draws from those global RNGs (dropout masks, weight
init-time-only randomness, etc.) continues on the same sequence it would
have followed without interruption. This does **not** make a resumed run
bit-identical to an uninterrupted one in every respect:

- The `DataLoader` iterator itself is rebuilt fresh on resume and is not
  fast-forwarded to the exact in-epoch position it was at — so the order of
  examples seen after a resume will differ from an uninterrupted run, even
  though the RNG stream driving `shuffle=True` is restored.
- With `num_workers > 0`, worker-process RNG seeding is not currently pinned
  to the checkpointed main-process state, so multi-worker data order is not
  exactly reproducible across resumes regardless of this RNG handling.

Treat resumed runs as "continues training validly from this point," not as
"bit-for-bit identical to never having stopped."
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch


CHECKPOINT_PATTERN = re.compile(r"ckpt_step(\d+)\.pt$")


def checkpoint_path(checkpoint_dir: str | Path, step: int) -> Path:
    return Path(checkpoint_dir) / f"ckpt_step{step}.pt"


def _capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    checkpoint_dir: str | Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any = None,
    extra: dict | None = None,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": _capture_rng_state(),
        "extra": extra or {},
    }
    path = checkpoint_path(checkpoint_dir, step)
    tmp_path = path.with_suffix(".pt.tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(path)  # atomic-ish on same filesystem; avoids corrupt partial reads
    return path


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    candidates = []
    for p in checkpoint_dir.glob("ckpt_step*.pt"):
        m = CHECKPOINT_PATTERN.search(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[-1][1]


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    map_location: str = "cpu",
    restore_rng: bool = True,
) -> dict:
    # weights_only=False: our checkpoints carry optimizer state and RNG
    # state (NumPy's RNG state tuple, Python's random state) alongside
    # tensors, which torch's default `weights_only=True` unpickler refuses
    # to deserialize. Only load checkpoints this pipeline produced itself
    # (e.g. from your own Drive-backed checkpoint_dir) — never a checkpoint
    # file from an untrusted source, since this path can execute arbitrary
    # code during unpickling.
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if restore_rng:
        _restore_rng_state(state.get("rng_state"))
    return state
