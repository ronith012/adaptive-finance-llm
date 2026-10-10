import pytest

from afllm.env_checks import StorageNotMountedError, verify_storage_path, verify_training_paths


def test_non_drive_path_is_not_checked(tmp_path, monkeypatch):
    # Path outside the mount root should never raise, regardless of mount state.
    monkeypatch.setattr("os.path.ismount", lambda _p: False)
    verify_storage_path(tmp_path / "checkpoints", mount_root="/content/drive")


def test_drive_path_raises_when_not_mounted(tmp_path, monkeypatch):
    fake_root = tmp_path / "content" / "drive"
    monkeypatch.setattr("os.path.ismount", lambda p: str(p) != str(fake_root))

    with pytest.raises(StorageNotMountedError):
        verify_storage_path(fake_root / "MyDrive" / "checkpoints", mount_root=fake_root)


def test_drive_path_passes_when_mounted(tmp_path, monkeypatch):
    fake_root = tmp_path / "content" / "drive"
    monkeypatch.setattr("os.path.ismount", lambda p: str(p) == str(fake_root))

    # Should not raise.
    verify_storage_path(fake_root / "MyDrive" / "checkpoints", mount_root=fake_root)


def test_verify_training_paths_checks_all_three(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "afllm.env_checks.verify_storage_path",
        lambda path, mount_root="/content/drive": calls.append(str(path)),
    )

    class FakeData:
        cache_dir = "/content/drive/MyDrive/data"

    class FakeTrain:
        checkpoint_dir = "/content/drive/MyDrive/checkpoints"
        log_dir = "/content/drive/MyDrive/logs"

    class FakeCfg:
        data = FakeData()
        train = FakeTrain()

    verify_training_paths(FakeCfg())
    assert calls == [
        "/content/drive/MyDrive/data",
        "/content/drive/MyDrive/checkpoints",
        "/content/drive/MyDrive/logs",
    ]
