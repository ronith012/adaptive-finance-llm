import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from smoke_test import run_smoke_test  # noqa: E402


def test_smoke_test_runs_on_cpu_with_local_checkpoint_dir():
    # No checkpoint_dir -> uses (and cleans up) a local temp dir; no network,
    # no Drive dependency. Should complete without raising.
    run_smoke_test(device="cpu", checkpoint_dir=None)


def test_smoke_test_runs_with_explicit_non_drive_checkpoint_dir(tmp_path):
    run_smoke_test(device="cpu", checkpoint_dir=str(tmp_path / "ckpt"))
    assert any((tmp_path / "ckpt").glob("ckpt_step*.pt"))
