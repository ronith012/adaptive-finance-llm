import pytest
import torch

from afllm.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig, TrainConfigError
from afllm.env_checks import CudaNotAvailableError
from afllm.train import EmptyTrainLoaderError, cosine_lr, resolve_amp_settings, train

LR = 3e-4
MIN_LR = 3e-5
WARMUP = 50
MAX_STEPS = 500


def test_warmup_reaches_lr_at_last_warmup_step():
    # Linear ramp: at step = warmup_steps - 1 (the last warmup step), the
    # schedule should have reached exactly `lr`.
    assert cosine_lr(WARMUP - 1, WARMUP, MAX_STEPS, LR, MIN_LR) == LR


def test_warmup_first_step_is_lr_over_warmup_steps():
    assert cosine_lr(0, WARMUP, MAX_STEPS, LR, MIN_LR) == LR / WARMUP


def test_warmup_to_cosine_handoff_is_continuous():
    # No discontinuity at the warmup/cosine boundary: both sides agree at lr.
    last_warmup = cosine_lr(WARMUP - 1, WARMUP, MAX_STEPS, LR, MIN_LR)
    first_cosine = cosine_lr(WARMUP, WARMUP, MAX_STEPS, LR, MIN_LR)
    assert last_warmup == LR
    assert first_cosine == LR
    assert last_warmup == first_cosine


def test_cosine_decays_toward_min_lr_by_last_step():
    last_step_lr = cosine_lr(MAX_STEPS - 1, WARMUP, MAX_STEPS, LR, MIN_LR)
    assert MIN_LR <= last_step_lr < LR
    # Should be very close to min_lr (cosine has nearly bottomed out).
    assert abs(last_step_lr - MIN_LR) < 1e-3 * LR


def test_lr_clamps_to_min_lr_at_and_past_max_steps():
    assert cosine_lr(MAX_STEPS, WARMUP, MAX_STEPS, LR, MIN_LR) == MIN_LR
    assert cosine_lr(MAX_STEPS + 100, WARMUP, MAX_STEPS, LR, MIN_LR) == MIN_LR


def test_lr_is_monotonic_non_increasing_after_warmup():
    prev = cosine_lr(WARMUP, WARMUP, MAX_STEPS, LR, MIN_LR)
    for step in range(WARMUP + 1, MAX_STEPS):
        current = cosine_lr(step, WARMUP, MAX_STEPS, LR, MIN_LR)
        assert current <= prev + 1e-12, f"lr increased at step {step}"
        prev = current


def test_cosine_lr_is_a_pure_function_of_step():
    # This is what makes resume preserve the intended LR schedule: the
    # same step must always produce the same lr, independent of call order
    # or any external state.
    for step in [0, 1, WARMUP - 1, WARMUP, WARMUP + 1, MAX_STEPS // 2, MAX_STEPS - 1]:
        a = cosine_lr(step, WARMUP, MAX_STEPS, LR, MIN_LR)
        b = cosine_lr(step, WARMUP, MAX_STEPS, LR, MIN_LR)
        assert a == b


def test_resolve_amp_settings_bare_cuda_enables_scaler_when_amp_requested():
    device_type, scaler_enabled = resolve_amp_settings("cuda", amp_requested=True)
    assert device_type == "cuda"
    assert scaler_enabled is True


def test_resolve_amp_settings_indexed_cuda_device_is_recognized_as_cuda():
    # This is the exact bug being fixed: "cuda:0" != "cuda" under plain
    # string equality, which previously left AMP silently disabled on an
    # indexed GPU device even with amp_requested=True.
    for indexed_device in ("cuda:0", "cuda:1", "cuda:7"):
        device_type, scaler_enabled = resolve_amp_settings(indexed_device, amp_requested=True)
        assert device_type == "cuda", f"{indexed_device!r} should resolve to device_type 'cuda'"
        assert scaler_enabled is True, f"{indexed_device!r} should enable the AMP scaler"


def test_resolve_amp_settings_cpu_always_disables_scaler():
    # Explicit CPU selection must never be second-guessed, regardless of
    # the amp flag's value.
    device_type, scaler_enabled = resolve_amp_settings("cpu", amp_requested=True)
    assert device_type == "cpu"
    assert scaler_enabled is False

    device_type, scaler_enabled = resolve_amp_settings("cpu", amp_requested=False)
    assert device_type == "cpu"
    assert scaler_enabled is False


def test_resolve_amp_settings_amp_false_disables_scaler_even_on_cuda():
    device_type, scaler_enabled = resolve_amp_settings("cuda:0", amp_requested=False)
    assert device_type == "cuda"
    assert scaler_enabled is False


def tiny_cfg(tmp_path, **train_overrides) -> ExperimentConfig:
    """A tiny, fast, CPU-only config for exercising train() directly, with
    no dataset download and no GPU required. checkpoint_dir/log_dir are
    local tmp_path subdirectories (not under /content/drive), so
    verify_training_paths' Drive-mount check never triggers."""
    model = ModelConfig(
        vocab_size=32, dim=16, n_layers=1, n_heads=2, n_kv_heads=2, max_seq_len=8,
    )
    train_cfg = TrainConfig(
        batch_size=4,
        grad_accum_steps=1,
        max_steps=1,
        lr=1e-3,
        min_lr=1e-4,
        warmup_steps=1,
        weight_decay=0.0,
        grad_clip=1e6,  # effectively disables clipping, isolates accumulation math
        eval_interval=1,
        eval_steps=1,
        checkpoint_interval=1,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
        seed=0,
        amp=False,
        device="cpu",
        resume=False,
    )
    for key, value in train_overrides.items():
        setattr(train_cfg, key, value)
    # Not under /content/drive, so verify_training_paths' Drive-mount
    # check (which also inspects data.cache_dir, not just checkpoint/log
    # dirs) never triggers on this dev machine.
    data = DataConfig(cache_dir=str(tmp_path / "data"))
    return ExperimentConfig(model=model, data=data, train=train_cfg)


def fixed_batch(seed: int, n_examples: int, seq_len: int, vocab_size: int):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, vocab_size, (n_examples, seq_len), generator=g)
    y = torch.randint(0, vocab_size, (n_examples, seq_len), generator=g)
    return x, y


def test_validate_train_config_is_enforced_by_train(tmp_path):
    cfg = tiny_cfg(tmp_path, eval_interval=0)
    x, y = fixed_batch(seed=1, n_examples=4, seq_len=8, vocab_size=32)
    loader = [(x, y)]
    with pytest.raises(TrainConfigError):
        train(cfg, loader, loader)


def test_resolve_device_is_enforced_by_train(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = tiny_cfg(tmp_path, device="cuda")
    x, y = fixed_batch(seed=1, n_examples=4, seq_len=8, vocab_size=32)
    loader = [(x, y)]
    with pytest.raises(CudaNotAvailableError):
        train(cfg, loader, loader)


def test_train_raises_clearly_on_empty_train_loader_with_len(tmp_path):
    cfg = tiny_cfg(tmp_path)
    empty_loader = []  # has __len__, equals 0
    val_x, val_y = fixed_batch(seed=2, n_examples=4, seq_len=8, vocab_size=32)
    with pytest.raises(EmptyTrainLoaderError):
        train(cfg, empty_loader, [(val_x, val_y)])


def test_train_raises_clearly_on_empty_iterable_train_loader_without_len(tmp_path):
    """An iterable with no __len__ at all (so the upfront len() check can't
    catch it) must still fail with a clear error from the inner retry
    guard, not a bare StopIteration."""

    class EmptyIterableNoLen:
        def __iter__(self):
            return iter([])

    cfg = tiny_cfg(tmp_path)
    val_x, val_y = fixed_batch(seed=2, n_examples=4, seq_len=8, vocab_size=32)
    with pytest.raises(EmptyTrainLoaderError):
        train(cfg, EmptyIterableNoLen(), [(val_x, val_y)])


def test_optimizer_step_called_exactly_once_per_outer_step(tmp_path, monkeypatch):
    step_calls = {"count": 0}
    real_step = torch.optim.AdamW.step

    def counting_step(self, *args, **kwargs):
        step_calls["count"] += 1
        return real_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", counting_step)

    cfg = tiny_cfg(tmp_path, grad_accum_steps=3, max_steps=2)
    x, y = fixed_batch(seed=3, n_examples=4, seq_len=8, vocab_size=32)
    # Enough micro-batches for 2 outer steps * 3 grad_accum_steps = 6 pulls,
    # recycled via the loader's re-iteration logic.
    loader = [(x, y)]

    train(cfg, loader, loader)

    assert step_calls["count"] == 2, (
        "optimizer.step() should run once per outer step regardless of "
        "grad_accum_steps, not once per micro-batch"
    )


def test_gradient_accumulation_matches_single_batch_equivalent(tmp_path):
    """The gold-standard check for grad-accumulation correctness: training
    on one combined batch of N examples in a single micro-step must
    produce (very close to) the same post-step weights as training on the
    same N examples split into grad_accum_steps equal micro-batches -
    provided each micro-batch has the same token count (true here: 2+2
    examples * 8 seq_len = 8 tokens each), since cross_entropy's mean
    reduction then makes the combined-batch loss exactly the average of
    the two micro-batch losses.
    """
    seq_len = 8
    vocab_size = 32
    x_all, y_all = fixed_batch(seed=42, n_examples=4, seq_len=seq_len, vocab_size=vocab_size)

    def run(grad_accum_steps: int, loader) -> dict:
        cfg = tiny_cfg(
            tmp_path / f"run_{grad_accum_steps}",
            grad_accum_steps=grad_accum_steps,
        )
        train(cfg, loader, loader)
        # Re-load the model this run just saved, to compare post-step
        # weights. optimizer=None: we only care about model weights here,
        # and build_optimizer's two param groups (decay/no_decay) wouldn't
        # match a plain single-group AdamW anyway.
        from afllm.checkpoint import find_latest_checkpoint, load_checkpoint
        from afllm.model import TransformerLM

        model = TransformerLM(cfg.model)
        ckpt = find_latest_checkpoint(cfg.train.checkpoint_dir)
        load_checkpoint(ckpt, model, optimizer=None, restore_rng=False)
        return {name: p.detach().clone() for name, p in model.named_parameters()}

    single_batch_loader = [(x_all, y_all)]
    weights_single = run(grad_accum_steps=1, loader=single_batch_loader)

    split_loader = [
        (x_all[:2], y_all[:2]),
        (x_all[2:], y_all[2:]),
    ]
    weights_split = run(grad_accum_steps=2, loader=split_loader)

    assert weights_single.keys() == weights_split.keys()
    for name in weights_single:
        assert torch.allclose(weights_single[name], weights_split[name], rtol=1e-4, atol=1e-6), (
            f"parameter {name!r} diverged between grad_accum_steps=1 and =2 "
            f"- gradient accumulation averaging is not behaving equivalently"
        )
