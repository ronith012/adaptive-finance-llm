import random

import numpy as np
import torch

from afllm.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
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
