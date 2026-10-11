"""Checkpoint save/load and recovery.

Checkpoints are plain `torch.save` dicts containing model, optimizer,
scheduler, AMP GradScaler, RNG, and step state, written to
`TrainConfig.checkpoint_dir` (a Google Drive path in the Colab workflow).
`find_latest_checkpoint` lets training resume after a Colab disconnect
without re-running from step 0.

GradScaler state (the AMP loss-scale value and its growth/backoff
bookkeeping) is saved/restored the same optional, additive way as
`scheduler`: pass `scaler=` to `save_checkpoint`/`load_checkpoint` and it
round-trips; omit it (or load an older checkpoint saved before this
existed) and nothing breaks - there is simply no scaler state to restore,
same as today's behavior.

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

Run metadata: `build_run_metadata` captures the full experiment config,
installed library versions, seed, and step into the checkpoint's `extra`
field (see `save_checkpoint`), so a checkpoint records what produced it.
`check_resume_compatibility` compares that saved config against the config
about to resume with, for the fields that actually affect reproducibility
(model architecture, dataset/tokenizer identity, seed, schedule, batching) —
cosmetic fields like `checkpoint_dir` or `log_dir` are deliberately excluded
since they legitimately differ across environments. It never blocks a
resume; callers (e.g. `train.py`) are expected to surface its result (via
logging) rather than silently proceeding as if an unverified or mismatched
resume were an exact continuation. A checkpoint saved before this feature
existed has no `config` key in `extra`, which is reported as its own
explicit status ("no_metadata") rather than conflated with a verified match.
"""

from __future__ import annotations

import dataclasses
import random
import re
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
from typing import Any

import numpy as np
import torch

from afllm.config import ExperimentConfig


CHECKPOINT_PATTERN = re.compile(r"ckpt_step(\d+)\.pt$")

# Config fields whose mismatch means a "resume" is not a true continuation
# of the same experiment. Deliberately excludes operational fields
# (checkpoint_dir, log_dir, num_workers, amp, device, resume, eval_*,
# checkpoint_interval) that can legitimately differ between environments
# or sessions without affecting what is actually being learned.
_CRITICAL_CONFIG_PATHS: tuple[tuple[str, str], ...] = (
    ("model", "vocab_size"),
    ("model", "dim"),
    ("model", "n_layers"),
    ("model", "n_heads"),
    ("model", "n_kv_heads"),
    ("model", "max_seq_len"),
    ("model", "dropout"),
    ("model", "rope_theta"),
    ("model", "tie_embeddings"),
    ("data", "dataset_name"),
    ("data", "dataset_config"),
    ("data", "tokenizer_name"),
    ("data", "seq_len"),
    ("train", "seed"),
    ("train", "lr"),
    ("train", "min_lr"),
    ("train", "warmup_steps"),
    ("train", "max_steps"),
    ("train", "grad_accum_steps"),
    ("train", "batch_size"),
    ("train", "weight_decay"),
    ("train", "grad_clip"),
)

_MISSING = "<missing>"


def capture_environment_versions() -> dict:
    """Best-effort snapshot of installed library versions relevant to
    reproducing a run. Libraries not installed in the current environment
    (e.g. transformers/datasets/tokenizers on a CPU-only dev machine that
    never calls `load_wikitext2`) are recorded as None rather than raising
    - this must be safe to call from any environment, not just Colab."""
    versions: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,  # None on a CPU-only build
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        versions["gpu_name"] = torch.cuda.get_device_name(0)

    for pkg in ("transformers", "datasets", "tokenizers"):
        try:
            versions[pkg] = _pkg_version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
    return versions


def build_run_metadata(cfg: ExperimentConfig, step: int) -> dict:
    """Metadata to pass as `save_checkpoint`'s `extra`, so a checkpoint
    records the full config, environment, seed, and step that produced it."""
    return {
        "config": dataclasses.asdict(cfg),
        "environment": capture_environment_versions(),
        "seed": cfg.train.seed,
        "step": step,
        "dataset_name": cfg.data.dataset_name,
        "dataset_config": cfg.data.dataset_config,
        "tokenizer_name": cfg.data.tokenizer_name,
    }


def _get_nested(d: dict, section: str, field: str) -> Any:
    return d.get(section, {}).get(field, _MISSING)


def diff_critical_config(saved_config: dict, current_cfg: ExperimentConfig) -> list[str]:
    """Compare a checkpoint's saved config dict against `current_cfg` for
    the fields in `_CRITICAL_CONFIG_PATHS`. Returns a human-readable
    description for each field that differs; an empty list means no
    critical-field mismatch was found."""
    current = dataclasses.asdict(current_cfg)
    mismatches = []
    for section, field in _CRITICAL_CONFIG_PATHS:
        saved_val = _get_nested(saved_config, section, field)
        current_val = _get_nested(current, section, field)
        if saved_val != current_val:
            mismatches.append(f"{section}.{field}: checkpoint={saved_val!r} vs current={current_val!r}")
    return mismatches


def check_resume_compatibility(saved_extra: dict | None, cfg: ExperimentConfig) -> dict:
    """Compare a loaded checkpoint's saved run metadata against `cfg`.

    Returns one of:
      {"status": "no_metadata"}                   - checkpoint predates
          this feature (no "config" key in its `extra`); compatibility
          cannot be verified one way or the other.
      {"status": "match"}                          - no critical-field
          difference found between the saved config and `cfg`.
      {"status": "mismatch", "mismatches": [...]}  - one or more critical
          fields differ; resuming will not be an exact continuation of the
          original experiment.

    Never raises and never blocks a resume - it is the caller's job (see
    `train.py`) to surface this result rather than silently proceeding as
    if an unverified or mismatched resume were a verified exact one.
    """
    saved_config = (saved_extra or {}).get("config")
    if saved_config is None:
        return {"status": "no_metadata"}
    mismatches = diff_critical_config(saved_config, cfg)
    if mismatches:
        return {"status": "mismatch", "mismatches": mismatches}
    return {"status": "match"}


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
        # torch.set_rng_state requires a CPU torch.ByteTensor regardless of
        # the device training runs on. If this checkpoint was loaded with
        # map_location pointed at a CUDA device, torch.load moves every
        # tensor in the pickle - including this one - onto that device,
        # which then makes this call raise
        # "TypeError: RNG state must be a torch.ByteTensor". Force it back
        # to CPU here so restoration works no matter what map_location the
        # caller used to load the rest of the checkpoint.
        torch.set_rng_state(state["torch_cpu"].cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        # Same reasoning as above, per-device: torch.cuda.set_rng_state_all
        # also expects each entry to be a CPU ByteTensor (it is itself
        # responsible for applying it to the right CUDA device).
        torch.cuda.set_rng_state_all([t.cpu() for t in state["torch_cuda"]])


def save_checkpoint(
    checkpoint_dir: str | Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    extra: dict | None = None,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        # GradScaler.state_dict() returns {} when the scaler is disabled
        # (e.g. CPU-only or amp=False) - saved as {} rather than None in
        # that case, which load_checkpoint treats as "nothing to restore".
        "scaler": scaler.state_dict() if scaler is not None else None,
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
    scaler: torch.cuda.amp.GradScaler | None = None,
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
    if scaler is not None and state.get("scaler"):
        # Truthy check (not "is not None"): GradScaler.state_dict() saves
        # {} when the scaler was disabled, and GradScaler.load_state_dict
        # on an empty dict is a no-op anyway - this just skips the call
        # entirely for clarity, and also naturally skips checkpoints saved
        # before scaler support existed (state.get("scaler") is None there).
        scaler.load_state_dict(state["scaler"])
    if restore_rng:
        _restore_rng_state(state.get("rng_state"))
    return state
