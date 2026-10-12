import json
import os
from pathlib import Path

import pytest
import torch

from afllm.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
    TrainConfigError,
    VocabSizeMismatchError,
)
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
        train(cfg, loader, loader, tokenizer_vocab_size=cfg.model.vocab_size)


def test_resolve_device_is_enforced_by_train(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = tiny_cfg(tmp_path, device="cuda")
    x, y = fixed_batch(seed=1, n_examples=4, seq_len=8, vocab_size=32)
    loader = [(x, y)]
    with pytest.raises(CudaNotAvailableError):
        train(cfg, loader, loader, tokenizer_vocab_size=cfg.model.vocab_size)


def test_train_raises_clearly_on_empty_train_loader_with_len(tmp_path):
    cfg = tiny_cfg(tmp_path)
    empty_loader = []  # has __len__, equals 0
    val_x, val_y = fixed_batch(seed=2, n_examples=4, seq_len=8, vocab_size=32)
    with pytest.raises(EmptyTrainLoaderError):
        train(cfg, empty_loader, [(val_x, val_y)], tokenizer_vocab_size=cfg.model.vocab_size)


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
        train(cfg, EmptyIterableNoLen(), [(val_x, val_y)], tokenizer_vocab_size=cfg.model.vocab_size)


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

    train(cfg, loader, loader, tokenizer_vocab_size=cfg.model.vocab_size)

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
        train(cfg, loader, loader, tokenizer_vocab_size=cfg.model.vocab_size)
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


def test_train_rejects_mismatched_tokenizer_vocab_size(tmp_path):
    cfg = tiny_cfg(tmp_path)
    x, y = fixed_batch(seed=5, n_examples=4, seq_len=8, vocab_size=32)
    loader = [(x, y)]
    with pytest.raises(VocabSizeMismatchError):
        train(cfg, loader, loader, tokenizer_vocab_size=cfg.model.vocab_size + 1)

    # No log files should have been written - the check fails before
    # log_dir's JSONL/run_metadata files are ever opened.
    log_dir = Path(cfg.train.log_dir)
    assert not log_dir.exists() or not any(log_dir.iterdir())


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _find_one(log_dir: Path, pattern: str) -> Path:
    matches = list(log_dir.glob(pattern))
    assert len(matches) == 1, f"expected exactly one {pattern!r} file, found {matches}"
    return matches[0]


def test_train_writes_well_formed_jsonl_log(tmp_path):
    cfg = tiny_cfg(tmp_path)
    x, y = fixed_batch(seed=6, n_examples=4, seq_len=8, vocab_size=32)
    loader = [(x, y)]

    train(cfg, loader, loader, tokenizer_vocab_size=cfg.model.vocab_size)

    log_dir = Path(cfg.train.log_dir)
    jsonl_path = _find_one(log_dir, "train_log_*.jsonl")
    records = _read_jsonl(jsonl_path)

    assert len(records) >= 1
    step_records = [r for r in records if "step" in r and "event" not in r]
    assert any(r["step"] == 0 for r in step_records)
    assert all("train_loss" in r and "lr" in r and "sec_per_step" in r for r in step_records)


def test_train_writes_run_metadata_on_success(tmp_path):
    cfg = tiny_cfg(tmp_path)
    x, y = fixed_batch(seed=7, n_examples=4, seq_len=8, vocab_size=32)
    loader = [(x, y)]

    train(cfg, loader, loader, tokenizer_vocab_size=cfg.model.vocab_size)

    log_dir = Path(cfg.train.log_dir)
    run_metadata_path = _find_one(log_dir, "run_metadata_*.json")
    metadata = json.loads(run_metadata_path.read_text())

    assert metadata["status"] == "completed"
    assert metadata["config"]["train"]["seed"] == cfg.train.seed
    assert metadata["seed"] == cfg.train.seed
    assert metadata["dataset_name"] == cfg.data.dataset_name
    assert "torch" in metadata["environment"]
    assert metadata["checkpoint_dir"] == cfg.train.checkpoint_dir
    assert metadata["final_step"] == 0  # tiny_cfg defaults to max_steps=1
    assert metadata["final_val_status"] == "ok"
    assert isinstance(metadata["duration_sec"], (int, float))
    assert metadata["duration_sec"] >= 0

    # The JSONL and run_metadata files share the same run_id, so they can
    # be paired up even if log_dir is reused across runs.
    jsonl_path = _find_one(log_dir, "train_log_*.jsonl")
    assert metadata["log_file"] == jsonl_path.name

    # tiny_cfg defaults to checkpoint_interval=1, so a real checkpoint was
    # written - last_checkpoint must name the actual file on disk, not
    # just the configured directory.
    from afllm.checkpoint import find_latest_checkpoint

    actual_latest = find_latest_checkpoint(cfg.train.checkpoint_dir)
    assert actual_latest is not None
    assert metadata["last_checkpoint"] == str(actual_latest)


def test_train_writes_failed_run_metadata_and_closes_log_on_exception(tmp_path):
    cfg = tiny_cfg(tmp_path)
    empty_loader = []
    val_x, val_y = fixed_batch(seed=8, n_examples=4, seq_len=8, vocab_size=32)

    with pytest.raises(EmptyTrainLoaderError):
        train(cfg, empty_loader, [(val_x, val_y)], tokenizer_vocab_size=cfg.model.vocab_size)

    log_dir = Path(cfg.train.log_dir)
    run_metadata_path = _find_one(log_dir, "run_metadata_*.json")
    metadata = json.loads(run_metadata_path.read_text())  # must parse - file was closed/flushed
    assert metadata["status"] == "failed"
    assert "EmptyTrainLoaderError" in metadata["error"]
    # No step ever ran, so there's nothing to call "ok" or
    # "no_validation_data" about - distinct from both of those.
    assert metadata["final_val_status"] == "not_evaluated"
    # No checkpoint was ever saved (failed before the first step).
    assert metadata["last_checkpoint"] is None

    # The JSONL file must exist and, even with zero records written before
    # the failure, still be a validly-closed (empty) file - not left open
    # or truncated mid-write.
    jsonl_path = _find_one(log_dir, "train_log_*.jsonl")
    assert _read_jsonl(jsonl_path) == []


def test_run_metadata_final_val_status_distinguishes_empty_validation_data(tmp_path):
    """Distinct from both "ok" (test_train_writes_run_metadata_on_success)
    and "not_evaluated" (the empty-train-loader failure test above): here
    training itself succeeds and an eval genuinely runs, but the
    validation set is empty, so evaluate() reports "valid": False."""
    cfg = tiny_cfg(tmp_path)
    x, y = fixed_batch(seed=9, n_examples=4, seq_len=8, vocab_size=32)
    train_loader = [(x, y)]
    empty_val_loader: list = []

    train(cfg, train_loader, empty_val_loader, tokenizer_vocab_size=cfg.model.vocab_size)

    log_dir = Path(cfg.train.log_dir)
    metadata = json.loads(_find_one(log_dir, "run_metadata_*.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["final_val_status"] == "no_validation_data"
    assert metadata["final_val_loss"] is None
    assert metadata["final_val_perplexity"] is None


def test_train_log_fn_keeps_its_original_positional_slot(tmp_path):
    """Regression test: tokenizer_vocab_size is keyword-only specifically
    so log_fn can still be passed positionally (as it always could, before
    tokenizer_vocab_size existed) without being misbound to the new
    parameter."""
    cfg = tiny_cfg(tmp_path)
    x, y = fixed_batch(seed=10, n_examples=4, seq_len=8, vocab_size=32)
    loader = [(x, y)]
    captured: list = []

    def my_log_fn(entry: dict) -> None:
        captured.append(entry)

    # log_fn passed positionally, exactly as any pre-existing caller would.
    train(cfg, loader, loader, my_log_fn, tokenizer_vocab_size=cfg.model.vocab_size)

    assert len(captured) >= 1
    assert all(isinstance(r, dict) for r in captured)
    assert any("step" in r for r in captured)


def test_train_tokenizer_vocab_size_cannot_be_passed_positionally(tmp_path):
    """tokenizer_vocab_size is keyword-only: a 5th positional argument
    must raise a clear TypeError rather than being silently accepted."""
    cfg = tiny_cfg(tmp_path)
    x, y = fixed_batch(seed=11, n_examples=4, seq_len=8, vocab_size=32)
    loader = [(x, y)]

    with pytest.raises(TypeError):
        train(cfg, loader, loader, print, cfg.model.vocab_size)  # 5 positional args


def test_train_jsonl_log_is_strict_json_even_with_nan_and_inf_loss(tmp_path, monkeypatch):
    """A diverging loss (a realistic occurrence) must not produce
    non-standard JSON (bare NaN/Infinity tokens). Forces model(x, y) to
    return a non-finite loss once, and verifies every JSONL line still
    parses under a strict reader that rejects bare NaN/Infinity/-Infinity
    constants - not just Python's own lenient json.loads."""
    import math

    from afllm.model import TransformerLM

    cfg = tiny_cfg(tmp_path)
    x, y = fixed_batch(seed=12, n_examples=4, seq_len=8, vocab_size=32)
    loader = [(x, y)]

    real_forward = TransformerLM.forward

    def forward_with_nan_loss(self, idx, targets=None):
        logits, loss = real_forward(self, idx, targets)
        if loss is not None:
            loss = loss * float("nan")
        return logits, loss

    monkeypatch.setattr(TransformerLM, "forward", forward_with_nan_loss)

    train(cfg, loader, loader, tokenizer_vocab_size=cfg.model.vocab_size)

    log_dir = Path(cfg.train.log_dir)
    jsonl_path = _find_one(log_dir, "train_log_*.jsonl")

    def reject_bare_constants(token: str):
        raise ValueError(f"non-standard JSON constant encountered: {token}")

    with open(jsonl_path) as f:
        lines = [line for line in f if line.strip()]
    assert len(lines) >= 1
    for line in lines:
        # Would raise if the line contained a bare NaN/Infinity/-Infinity
        # token; succeeds here because _json_safe already converted any
        # such float to the ordinary JSON string "NaN"/"Infinity".
        record = json.loads(line, parse_constant=reject_bare_constants)
        if "train_loss" in record:
            assert record["train_loss"] == "NaN"


def test_same_second_run_ids_do_not_collide_or_mix_runs(tmp_path, monkeypatch):
    """Two train() calls forced to share an identical timestamp prefix
    (simulating two invocations within the same wall-clock second) must
    still produce two distinct, uncorrupted sets of log files."""
    import afllm.train as train_module

    monkeypatch.setattr(train_module, "_utc_timestamp", lambda: "20260101T000000Z")

    cfg = tiny_cfg(tmp_path)
    x1, y1 = fixed_batch(seed=13, n_examples=4, seq_len=8, vocab_size=32)
    x2, y2 = fixed_batch(seed=14, n_examples=4, seq_len=8, vocab_size=32)

    train(cfg, [(x1, y1)], [(x1, y1)], tokenizer_vocab_size=cfg.model.vocab_size)
    train(cfg, [(x2, y2)], [(x2, y2)], tokenizer_vocab_size=cfg.model.vocab_size)

    log_dir = Path(cfg.train.log_dir)
    jsonl_files = sorted(log_dir.glob("train_log_*.jsonl"))
    metadata_files = sorted(log_dir.glob("run_metadata_*.json"))

    assert len(jsonl_files) == 2, f"expected 2 distinct jsonl files, found {jsonl_files}"
    assert len(metadata_files) == 2, f"expected 2 distinct run_metadata files, found {metadata_files}"
    assert jsonl_files[0] != jsonl_files[1]

    # Neither run's single step-record got mixed into the other's file.
    for jsonl_path in jsonl_files:
        records = _read_jsonl(jsonl_path)
        step_records = [r for r in records if "step" in r and "event" not in r]
        assert len(step_records) == 1, (
            f"{jsonl_path} should contain exactly this one run's record, got {records}"
        )

    for metadata_path in metadata_files:
        metadata = json.loads(metadata_path.read_text())
        assert metadata["status"] == "completed"


def test_train_metadata_write_failure_after_success_is_reported_distinctly(tmp_path, monkeypatch):
    """If training itself fully succeeds but persisting the final
    run_metadata.json then fails, train() must raise RunMetadataWriteError
    (never claim training failed, never return a misleading status)."""
    import afllm.train as train_module
    from afllm.train import RunMetadataWriteError

    real_atomic_write = train_module._atomic_write_json

    def failing_on_final_write(path, data, *, exclusive=False):
        if data.get("status") == "completed":
            raise OSError("simulated disk-full on final metadata write")
        real_atomic_write(path, data, exclusive=exclusive)

    monkeypatch.setattr(train_module, "_atomic_write_json", failing_on_final_write)

    cfg = tiny_cfg(tmp_path)
    x, y = fixed_batch(seed=15, n_examples=4, seq_len=8, vocab_size=32)
    loader = [(x, y)]

    with pytest.raises(RunMetadataWriteError) as exc_info:
        train(cfg, loader, loader, tokenizer_vocab_size=cfg.model.vocab_size)

    message = str(exc_info.value)
    assert "completed successfully" in message
    assert "did NOT fail" in message
    # Training genuinely ran: a checkpoint exists even though this call
    # raised, proving the raised error is about metadata, not training.
    from afllm.checkpoint import find_latest_checkpoint

    assert find_latest_checkpoint(cfg.train.checkpoint_dir) is not None


def test_train_interrupted_run_is_marked_interrupted_not_completed(tmp_path, monkeypatch):
    """A deterministic stand-in for Ctrl-C mid-training: the exception
    must propagate unchanged, the JSONL file must still close safely, and
    run_metadata must say "interrupted" - never "completed" or "failed"."""

    def interrupting_step(self, *args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(torch.optim.AdamW, "step", interrupting_step)

    cfg = tiny_cfg(tmp_path)
    x, y = fixed_batch(seed=16, n_examples=4, seq_len=8, vocab_size=32)
    loader = [(x, y)]

    with pytest.raises(KeyboardInterrupt):
        train(cfg, loader, loader, tokenizer_vocab_size=cfg.model.vocab_size)

    log_dir = Path(cfg.train.log_dir)
    metadata = json.loads(_find_one(log_dir, "run_metadata_*.json").read_text())
    assert metadata["status"] == "interrupted"

    # The jsonl file must still be a well-formed, closed file.
    jsonl_path = _find_one(log_dir, "train_log_*.jsonl")
    assert _read_jsonl(jsonl_path) == []


def test_train_both_training_and_metadata_write_fail_preserve_original_exception(
    tmp_path, monkeypatch
):
    """Training fails AND the final run_metadata write also fails. The
    original training exception - a specific instance, not just a type -
    must still be what propagates to the caller; the metadata-write
    failure must be surfaced separately through log_fn (the same
    caller-supplied reporting channel train() already uses for every
    other event); and the JSONL file must still be closed and readable.

    Avoids monkeypatching train.py's own private helpers: the training
    failure is forced through torch.optim.AdamW.step (a third-party hook
    already used by other tests in this file), and the metadata-write
    failure is forced through the standard library's os.replace - public
    API surface, not an afllm-internal name - failing only on the second
    call targeting this run's metadata path (the first, initial "running"
    write must still succeed so training actually proceeds).
    """
    specific_exception = RuntimeError("deliberately forced training failure, instance X")

    def raising_step(self, *args, **kwargs):
        raise specific_exception

    monkeypatch.setattr(torch.optim.AdamW, "step", raising_step)

    # The run's first ("running") metadata write uses os.link, not
    # os.replace (see _atomic_write_json's exclusive=True path) - so the
    # only os.replace call targeting a run_metadata .json file in this
    # scenario is the final ("failed") write. Fail it unconditionally.
    real_os_replace = os.replace

    def flaky_replace(src, dst):
        if str(dst).endswith(".json"):
            raise OSError("simulated disk-full on final metadata write")
        return real_os_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky_replace)

    cfg = tiny_cfg(tmp_path)
    x, y = fixed_batch(seed=17, n_examples=4, seq_len=8, vocab_size=32)
    loader = [(x, y)]
    captured_logs: list = []

    with pytest.raises(RuntimeError) as exc_info:
        train(cfg, loader, loader, captured_logs.append, tokenizer_vocab_size=cfg.model.vocab_size)

    # The exact original exception instance must be what propagated - not
    # a new exception, not one replaced by the metadata-write failure.
    assert exc_info.value is specific_exception

    # The metadata-write failure must be reported through the intended
    # channel (log_fn) separately from the training outcome, not silently
    # dropped.
    metadata_failure_events = [
        entry for entry in captured_logs
        if isinstance(entry, dict) and entry.get("event") == "run_metadata_write_failed"
    ]
    assert len(metadata_failure_events) == 1
    assert "simulated disk-full" in metadata_failure_events[0]["error"]
    assert metadata_failure_events[0]["training_status"] == "failed"

    # The JSONL file must still be closed and readable (no step records
    # expected: the forced failure happens on the very first optimizer
    # step, before any step is ever logged).
    log_dir = Path(cfg.train.log_dir)
    jsonl_path = _find_one(log_dir, "train_log_*.jsonl")
    assert _read_jsonl(jsonl_path) == []


def test_train_does_not_overwrite_stale_run_metadata_lacking_a_jsonl_file(tmp_path, monkeypatch):
    """A pre-existing run_metadata_<id>.json with no matching
    train_log_<id>.jsonl (e.g. left over from an unrelated earlier
    attempt that reused the same log_dir) must never be silently
    overwritten just because a new run happens to generate that same id."""
    import afllm.train as train_module

    fixed_run_id = "20260101T000000Z_deadbeef"
    monkeypatch.setattr(train_module, "_new_run_id", lambda: fixed_run_id)

    cfg = tiny_cfg(tmp_path)
    log_dir = Path(cfg.train.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    stale_metadata_path = log_dir / f"run_metadata_{fixed_run_id}.json"
    stale_content = {"status": "completed", "note": "stale metadata from an unrelated earlier attempt"}
    stale_metadata_path.write_text(json.dumps(stale_content))
    # Deliberately no matching train_log_<fixed_run_id>.jsonl file.

    x, y = fixed_batch(seed=18, n_examples=4, seq_len=8, vocab_size=32)
    loader = [(x, y)]

    with pytest.raises(FileExistsError):
        train(cfg, loader, loader, tokenizer_vocab_size=cfg.model.vocab_size)

    # The stale file must be completely untouched.
    assert json.loads(stale_metadata_path.read_text()) == stale_content
    # No jsonl file (or stray temp file) should have been created for this
    # collided run id - the collision is caught before it's touched.
    assert not (log_dir / f"train_log_{fixed_run_id}.jsonl").exists()
    assert list(log_dir.glob("*.tmp-*")) == []
