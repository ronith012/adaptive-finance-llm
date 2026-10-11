import random

import numpy as np
import pytest
import torch

from afllm.checkpoint import (
    _restore_rng_state,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from afllm.config import ModelConfig
from afllm.model import TransformerLM


def tiny_model() -> TransformerLM:
    cfg = ModelConfig(
        vocab_size=97, dim=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=16
    )
    return TransformerLM(cfg)


def test_save_and_load_round_trip(tmp_path):
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    path = save_checkpoint(tmp_path, step=5, model=model, optimizer=optimizer)
    assert path.exists()

    model2 = tiny_model()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    state = load_checkpoint(path, model2, optimizer2)

    assert state["step"] == 5
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)


def test_find_latest_checkpoint_picks_highest_step(tmp_path):
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    save_checkpoint(tmp_path, step=0, model=model, optimizer=optimizer)
    save_checkpoint(tmp_path, step=50, model=model, optimizer=optimizer)
    save_checkpoint(tmp_path, step=10, model=model, optimizer=optimizer)

    latest = find_latest_checkpoint(tmp_path)
    assert latest is not None
    assert "step50" in latest.name


def test_find_latest_checkpoint_empty_dir_returns_none(tmp_path):
    assert find_latest_checkpoint(tmp_path / "does_not_exist") is None


def test_rng_state_round_trips_and_continues_sequence(tmp_path):
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

    # Burn some draws so state is mid-sequence, not a fresh seed.
    for _ in range(5):
        random.random()
        np.random.random()
        torch.rand(1)

    path = save_checkpoint(tmp_path, step=0, model=model, optimizer=optimizer)

    # Record what the next draws *should* be post-checkpoint.
    expected_py = random.random()
    expected_np = np.random.random()
    expected_torch = torch.rand(1)

    # Perturb global RNG state to simulate a different run having happened.
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    for _ in range(3):
        random.random()
        np.random.random()
        torch.rand(1)

    load_checkpoint(path, model, optimizer, restore_rng=True)

    assert random.random() == expected_py
    assert np.random.random() == expected_np
    assert torch.equal(torch.rand(1), expected_torch)


def test_load_checkpoint_can_skip_rng_restore(tmp_path):
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = save_checkpoint(tmp_path, step=0, model=model, optimizer=optimizer)

    random.seed(42)
    before = random.random()
    random.seed(42)  # reset so the next draw would match `before` if untouched

    load_checkpoint(path, model, optimizer, restore_rng=False)
    after = random.random()

    # With restore_rng=False, the checkpoint's saved RNG state must not be
    # applied — `after` should just be whatever the un-restored stream gives,
    # which here (same seed reset) equals `before`.
    assert after == before


def test_restore_rng_state_restores_correct_sequence_via_cpu_normalization():
    """Regression test for the Colab GPU smoke-test failure:
    `TypeError: RNG state must be a torch.ByteTensor`.

    When a checkpoint is loaded with `map_location="cuda"`, torch.load moves
    *every* tensor in the pickle onto that device, including the saved RNG
    state tensor — but torch.set_rng_state requires a CPU tensor regardless
    of what device training runs on.

    This doesn't need real CUDA hardware to exercise the actual bug trigger:
    `RequiresCpuNormalization` below is deliberately NOT a torch.Tensor, so
    passing it straight to the real, unmocked `torch.set_rng_state` would
    raise exactly that TypeError unless the code normalizes it via `.cpu()`
    first (confirmed separately: `torch.set_rng_state(wrapper)` without the
    `.cpu()` call raises `TypeError: expected a torch.ByteTensor, but got
    RequiresCpuNormalization`). Its `.cpu()` returns a real, previously
    captured CPU ByteTensor, so this test verifies actual behavior — the
    real RNG draw sequence after restoration — not merely that `.cpu()` was
    called on something. Nothing here is mocked: `_restore_rng_state` and
    `torch.set_rng_state` both run for real.
    """

    class RequiresCpuNormalization:
        """Stands in for a tensor that map_location moved onto a non-CPU
        device. Passing this directly to torch.set_rng_state fails because
        it isn't a real tensor; `.cpu()` is what must be called to recover
        the usable CPU ByteTensor, exactly mirroring the real fix."""

        def __init__(self, real_cpu_tensor: torch.Tensor):
            self._real_cpu_tensor = real_cpu_tensor

        def cpu(self) -> torch.Tensor:
            return self._real_cpu_tensor

    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)
    for _ in range(5):
        random.random()
        np.random.random()
        torch.rand(1)

    # A real CPU ByteTensor, captured mid-sequence.
    captured_cpu_state = torch.get_rng_state()
    # What the RNG should produce next if this exact state is restored.
    expected_next_draw = torch.rand(1)

    # Perturb global torch RNG state to simulate a different run having
    # happened in between (e.g. a Colab session continuing after save).
    torch.manual_seed(999)
    for _ in range(3):
        torch.rand(1)

    # Real call, nothing mocked — this is exactly what load_checkpoint does
    # internally, but with the CPU state deliberately wrapped so it is not
    # already a plain CPU tensor when handed in.
    _restore_rng_state({"torch_cpu": RequiresCpuNormalization(captured_cpu_state)})

    actual_next_draw = torch.rand(1)
    assert torch.equal(actual_next_draw, expected_next_draw), (
        "restoring RNG state through a non-tensor wrapper did not reproduce "
        "the expected draw sequence - .cpu() normalization regressed"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires a real CUDA device to reproduce map_location='cuda' tensor placement",
)
def test_load_checkpoint_with_map_location_cuda_round_trips(tmp_path):
    """End-to-end reproduction of the reported bug on real GPU hardware.
    Skipped (not failed) on machines without CUDA, such as this dev Mac —
    run on a GPU runtime (e.g. Colab) to exercise it for real."""
    model = tiny_model().to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)

    path = save_checkpoint(tmp_path, step=0, model=model, optimizer=optimizer)

    model2 = tiny_model().to("cuda")
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    # This is exactly the reported failure mode: map_location="cuda" moves
    # every tensor in the checkpoint, including the RNG state, onto CUDA.
    state = load_checkpoint(path, model2, optimizer2, map_location="cuda")

    assert state["step"] == 0
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)


def test_load_checkpoint_backward_compatible_without_rng_state(tmp_path):
    """A checkpoint saved before RNG-state capture existed (no "rng_state"
    key at all) must still load without raising."""
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    legacy_state = {
        "step": 3,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None,
        "extra": {},
    }
    path = tmp_path / "ckpt_step3.pt"
    torch.save(legacy_state, path)

    model2 = tiny_model()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    state = load_checkpoint(path, model2, optimizer2)  # must not raise

    assert state["step"] == 3


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GradScaler forcibly disables itself (state_dict() == {}) when CUDA "
    "is unavailable, regardless of enabled=True, so a non-trivial enabled "
    "scaler state can only be exercised on real CUDA hardware",
)
def test_scaler_state_round_trips(tmp_path):
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Non-default construction args so this scaler's state is clearly
    # different from a freshly-constructed default GradScaler's state -
    # proves restoration actually happened, not just "defaults match
    # defaults".
    scaler = torch.cuda.amp.GradScaler(
        init_scale=2.0 ** 5, growth_factor=1.5, backoff_factor=0.25,
        growth_interval=50, enabled=True,
    )
    original_state = scaler.state_dict()

    path = save_checkpoint(tmp_path, step=0, model=model, optimizer=optimizer, scaler=scaler)

    model2 = tiny_model()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    scaler2 = torch.cuda.amp.GradScaler(enabled=True)  # plain defaults
    assert scaler2.state_dict() != original_state, (
        "test setup invalid: default scaler already matches the non-default one"
    )

    load_checkpoint(path, model2, optimizer2, scaler=scaler2)

    assert scaler2.state_dict() == original_state


def test_scaler_disabled_round_trip_is_a_no_op(tmp_path):
    """On CPU-only training, the scaler is constructed with enabled=False;
    its state_dict() is {} and restoring it must not raise or do anything
    surprising."""
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    path = save_checkpoint(tmp_path, step=0, model=model, optimizer=optimizer, scaler=scaler)

    model2 = tiny_model()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    scaler2 = torch.cuda.amp.GradScaler(enabled=False)

    load_checkpoint(path, model2, optimizer2, scaler=scaler2)  # must not raise
    assert scaler2.state_dict() == {}


def test_load_checkpoint_backward_compatible_without_scaler_key(tmp_path):
    """A checkpoint saved before scaler-state capture existed (no "scaler"
    key at all) must still load without raising, even when a real scaler
    is passed in to be restored."""
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    legacy_state = {
        "step": 4,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None,
        "extra": {},
    }
    path = tmp_path / "ckpt_step4.pt"
    torch.save(legacy_state, path)

    model2 = tiny_model()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    scaler2 = torch.cuda.amp.GradScaler(
        init_scale=2.0 ** 5, growth_factor=1.5, backoff_factor=0.25,
        growth_interval=50, enabled=True,
    )
    pre_load_state = scaler2.state_dict()

    state = load_checkpoint(path, model2, optimizer2, scaler=scaler2)  # must not raise

    assert state["step"] == 4
    # Nothing to restore from this legacy checkpoint - scaler2 untouched.
    assert scaler2.state_dict() == pre_load_state
