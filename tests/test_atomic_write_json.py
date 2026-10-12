import json
import os
import stat

import pytest

from afllm.train import _atomic_write_json


def test_atomic_write_json_round_trips(tmp_path):
    path = tmp_path / "data.json"
    _atomic_write_json(path, {"a": 1, "b": [1, 2, 3]})

    assert json.loads(path.read_text()) == {"a": 1, "b": [1, 2, 3]}
    # No leftover temp file.
    assert list(tmp_path.glob("*.tmp-*")) == []


def test_atomic_write_json_overwrite_leaves_no_temp_file(tmp_path):
    path = tmp_path / "data.json"
    _atomic_write_json(path, {"status": "running"})
    _atomic_write_json(path, {"status": "completed"})

    assert json.loads(path.read_text()) == {"status": "completed"}
    assert list(tmp_path.glob("*.tmp-*")) == []


def test_atomic_write_json_failure_preserves_previously_valid_file(tmp_path):
    """A real (not mocked) OS-level failure: make the directory read-only
    so creating the temp file fails, and confirm the file that was already
    there - written by an earlier, successful call - is untouched."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("running as root bypasses directory write permissions")

    path = tmp_path / "run_metadata.json"
    _atomic_write_json(path, {"status": "running", "step": 0})
    original_content = path.read_text()

    os.chmod(tmp_path, stat.S_IREAD | stat.S_IEXEC)  # read-only directory
    try:
        try:
            _atomic_write_json(path, {"status": "completed", "step": 99})
            raised = False
        except OSError:
            raised = True
    finally:
        os.chmod(tmp_path, stat.S_IRWXU)  # restore so pytest can clean up tmp_path

    assert raised, "expected the write to a read-only directory to fail"
    assert path.read_text() == original_content, (
        "a failed write must never destroy the previously valid file"
    )
    assert list(tmp_path.glob("*.tmp-*")) == [], "no stray temp file should remain"


def test_atomic_write_json_exclusive_succeeds_when_path_is_new(tmp_path):
    path = tmp_path / "data.json"
    _atomic_write_json(path, {"status": "running"}, exclusive=True)

    assert json.loads(path.read_text()) == {"status": "running"}
    assert list(tmp_path.glob("*.tmp-*")) == []


def test_atomic_write_json_exclusive_refuses_to_overwrite_existing_file(tmp_path):
    path = tmp_path / "data.json"
    _atomic_write_json(path, {"status": "stale", "from": "an earlier run"})

    with pytest.raises(FileExistsError):
        _atomic_write_json(path, {"status": "running", "from": "a new run"}, exclusive=True)

    # The pre-existing file must be completely untouched.
    assert json.loads(path.read_text()) == {"status": "stale", "from": "an earlier run"}
    assert list(tmp_path.glob("*.tmp-*")) == []


def test_atomic_write_json_non_exclusive_still_overwrites_same_run(tmp_path):
    """exclusive=True is only for a run's first write - subsequent updates
    to the same run's file must still use plain atomic replace."""
    path = tmp_path / "data.json"
    _atomic_write_json(path, {"status": "running"}, exclusive=True)
    _atomic_write_json(path, {"status": "completed"})  # same run, no exclusive=

    assert json.loads(path.read_text()) == {"status": "completed"}


def test_atomic_write_json_exclusive_does_not_use_hard_links(tmp_path):
    """Regression test for the fix itself: exclusive creation must not
    depend on hard links (os.link), which are not reliably supported on
    the FUSE-backed Google Drive mount this project's Colab training
    workflow relies on. Observable proxy: a destination whose directory
    entry count can only ever be 1 (a hard link would briefly or
    permanently produce a second directory entry for the same inode
    elsewhere) - inode link count stays at 1 after an exclusive write."""
    path = tmp_path / "data.json"
    _atomic_write_json(path, {"status": "running"}, exclusive=True)

    assert path.stat().st_nlink == 1


def test_atomic_write_json_exclusive_failure_partway_does_not_silently_succeed(
    tmp_path, monkeypatch
):
    """Documented tradeoff of the exclusive path: it writes directly into
    the destination (no temp-file swap), so a failure partway through
    can't be fully rolled back - but it must never be silently treated as
    a successful write. A fsync failure here is a stand-in for a real
    write interruption (e.g. a dropped connection to a network-mounted
    log_dir)."""
    path = tmp_path / "data.json"

    def failing_fsync(fd):
        raise OSError("simulated write interruption")

    monkeypatch.setattr(os, "fsync", failing_fsync)

    with pytest.raises(OSError):
        _atomic_write_json(path, {"status": "running"}, exclusive=True)
