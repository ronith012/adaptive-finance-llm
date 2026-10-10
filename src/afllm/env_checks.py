"""Startup safety checks for the training environment.

These exist to fail loudly instead of silently: the training config points
checkpoint/log/data paths at a Google Drive mount (e.g.
`/content/drive/MyDrive/...`) so that work survives a Colab runtime
recycling. If Drive isn't actually mounted, `Path.mkdir(parents=True)` will
happily create that same-looking path on the ephemeral local Colab disk
instead — training appears to work, and everything is lost on disconnect.
"""

from __future__ import annotations

import os
from pathlib import Path

DRIVE_MOUNT_ROOT = "/content/drive"


class StorageNotMountedError(RuntimeError):
    pass


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return path == root


def verify_storage_path(path: str | Path, mount_root: str | Path = DRIVE_MOUNT_ROOT) -> None:
    """Raise `StorageNotMountedError` if `path` is configured under
    `mount_root` but `mount_root` is not currently a mounted filesystem.

    Paths outside `mount_root` (e.g. local test paths, a `tmp_path` in
    tests, or an intentionally local run) are left alone — this check is
    specifically about avoiding silent writes to ephemeral storage when
    Drive-backed persistence was configured.
    """
    path = Path(path)
    mount_root = Path(mount_root)

    if not _is_under(path, mount_root):
        return

    if not os.path.ismount(mount_root):
        raise StorageNotMountedError(
            f"Configured path '{path}' is under '{mount_root}', but "
            f"'{mount_root}' is not currently mounted. This usually means "
            f"Google Drive was not mounted (or was disconnected) in this "
            f"Colab session. Run:\n\n"
            f"    from google.colab import drive\n"
            f"    drive.mount('{mount_root}')\n\n"
            f"before training — otherwise checkpoints, logs, and the "
            f"dataset cache will be silently written to ephemeral local "
            f"storage and lost when the runtime recycles."
        )


def verify_training_paths(cfg) -> None:
    """Verify all Drive-backed paths in an ExperimentConfig are mounted.

    Accepts a loosely-typed `cfg` (duck-typed on `.data.cache_dir`,
    `.train.checkpoint_dir`, `.train.log_dir`) to avoid a circular import
    with `afllm.config`.
    """
    verify_storage_path(cfg.data.cache_dir)
    verify_storage_path(cfg.train.checkpoint_dir)
    verify_storage_path(cfg.train.log_dir)
