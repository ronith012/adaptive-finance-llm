import pytest
import torch

from afllm.env_checks import (
    CudaNotAvailableError,
    StorageNotMountedError,
    resolve_device,
    verify_storage_path,
    verify_training_paths,
)


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


def test_resolve_device_cpu_is_always_honored(monkeypatch):
    # "cpu" is never second-guessed, regardless of CUDA availability.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cpu") == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_cuda_requested_but_unavailable_raises_clearly(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(CudaNotAvailableError, match="cuda"):
        resolve_device("cuda")


def test_resolve_device_cuda_requested_and_available_is_honored(monkeypatch):
    # Doesn't require real GPU hardware: only the availability check is
    # mocked, and resolve_device must return the request unchanged when it
    # reports True.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("cuda:0") == "cuda:0"


def test_resolve_device_real_hardware_check_on_this_machine():
    # No mocking - exercises the real torch.cuda.is_available() on whatever
    # machine runs this test. On this CPU-only dev environment, "cuda"
    # must raise; "cpu" must always pass.
    assert resolve_device("cpu") == "cpu"
    if torch.cuda.is_available():
        assert resolve_device("cuda") == "cuda"
    else:
        with pytest.raises(CudaNotAvailableError):
            resolve_device("cuda")
