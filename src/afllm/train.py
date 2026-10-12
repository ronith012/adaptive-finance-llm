"""Training loop with grad accumulation, AMP, logging, and checkpoint recovery."""

from __future__ import annotations

import json
import math
import os
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from afllm.checkpoint import (
    build_run_metadata,
    check_resume_compatibility,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from afllm.config import ExperimentConfig, validate_train_config, validate_vocab_size
from afllm.env_checks import resolve_device, verify_training_paths
from afllm.evaluate import evaluate
from afllm.model import TransformerLM


class EmptyTrainLoaderError(RuntimeError):
    pass


class RunMetadataWriteError(RuntimeError):
    """Raised only when training itself completed successfully but
    persisting the final run_metadata.json failed - this is always a
    distinct condition from a training failure, never a substitute for
    one. See `train()`'s final metadata-write handling."""

    pass


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _new_run_id() -> str:
    """A collision-resistant id for one run's log files: a human-sortable
    UTC timestamp plus a random suffix, so train_log_<id>.jsonl and
    run_metadata_<id>.json can be paired up, and two invocations against
    the same log_dir within the same second never collide."""
    return f"{_utc_timestamp()}_{uuid.uuid4().hex[:8]}"


def _json_safe(obj):
    """Recursively replace NaN/+-Infinity floats with their standard
    string names. Python's own `json.dumps` happily emits bare NaN/
    Infinity/-Infinity tokens, which round-trip through `json.loads` but
    are not valid per RFC 8259 and will break a strict external JSON
    parser - a realistic risk here, since a diverging loss is a normal
    thing to hit while iterating on a from-scratch model."""
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _dumps_strict(data) -> str:
    # allow_nan=False: a safety net. If a NaN/Infinity somehow reaches
    # this call un-sanitized (e.g. a future caller forgets _json_safe),
    # raise loudly rather than silently emit non-standard JSON.
    return json.dumps(_json_safe(data), default=str, allow_nan=False)


def _atomic_write_json(path: Path, data: dict, *, exclusive: bool = False) -> None:
    """Write `data` as JSON to `path`.

    `exclusive=True` (a run's first, "running" write - the one that
    actually claims its run id) creates `path` via plain `open(path, "x")`:
    POSIX's `O_CREAT | O_EXCL` makes the existence check and the creation
    one atomic kernel operation, so this raises `FileExistsError` (leaving
    `path` completely untouched) if a file is already there, rather than a
    separate `path.exists()` check followed by a write, which would have a
    check-then-write race. This is deliberately NOT implemented with
    `os.link` (hard links): hard links are not reliably supported on the
    FUSE-backed Google Drive mount this project's Colab training workflow
    depends on (`train.checkpoint_dir`/`log_dir` point under
    `/content/drive/...` by default - see `configs/baseline_small.yaml`),
    where they can fail outright. Plain exclusive file creation is the
    same mechanism this function already relied on for the JSONL log
    (`open(jsonl_path, "x")` in `train()`), so it carries no additional
    portability risk beyond what this codebase already depends on. The
    tradeoff: content is written directly into `path` rather than via a
    temp-file swap, so a crash partway through this specific write can
    leave a partially-written file at `path` - acceptable here because
    this is `path`'s very first write, so there was no previously-valid
    content to lose; the collision guarantee itself (this run id was not
    already claimed) remains fully atomic regardless of how far the write
    itself got.

    `exclusive=False` (the default - every later update to the *same*
    run) keeps a write-to-temp-file-then-replace pattern for full
    crash-safety: write to a uniquely-named sibling temp file, flush and
    fsync it, then `os.replace` it onto `path`. A reader never observes a
    partially-written file, and if the write fails before the replace,
    `path`'s previous (valid) contents are left completely untouched.

    Limitations (both modes): neither is atomic across filesystem
    boundaries (e.g. a network-mounted log_dir backed by a different
    filesystem than a temp file would be); and neither can protect
    against a hard kill (SIGKILL) landing before the write/replace
    completes - no userspace scheme can guarantee that.
    """
    if exclusive:
        with open(path, "x") as f:
            f.write(_dumps_strict(data))
            f.flush()
            os.fsync(f.fileno())
        return

    tmp_path = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    try:
        with open(tmp_path, "w") as f:
            f.write(_dumps_strict(data))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def cosine_lr(step: int, warmup_steps: int, max_steps: int, lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


def resolve_amp_settings(device: str, amp_requested: bool) -> tuple[str, bool]:
    """Returns (device_type, scaler_enabled) for a resolved device string
    and the config's `amp` flag.

    Uses `torch.device(device).type` - PyTorch's own parsing - rather than
    comparing `device` against the literal string "cuda". That string
    comparison previously left AMP silently disabled (and autocast
    silently configured for "cpu") whenever `device` named a specific GPU
    index like "cuda:0" rather than bare "cuda": `"cuda:0" == "cuda"` is
    False, even though it's very much a CUDA device.
    `torch.device("cuda:0").type` correctly returns "cuda" regardless of
    index, and `torch.device("cpu").type` returns "cpu" - CPU-only
    operation is unaffected.
    """
    device_type = torch.device(device).type
    scaler_enabled = amp_requested and device_type == "cuda"
    return device_type, scaler_enabled


def train(
    cfg: ExperimentConfig,
    train_loader,
    val_loader,
    log_fn=print,
    *,
    tokenizer_vocab_size: int,
) -> dict:
    """Run training. `log_fn` receives a dict per logged step (so callers
    can route logs to a file, Colab cell output, etc) - unchanged in
    position and behavior from before this function gained logging/vocab
    checking. Every record passed to `log_fn` is also appended as one
    strict-JSON line to a `train_log_<run_id>.jsonl` file under
    `cfg.train.log_dir`, and a paired `run_metadata_<run_id>.json` records
    this run's config, seed, environment, dataset identity, duration,
    final validation status/metrics, and the actual last checkpoint file
    written - written once at run start (`status: "running"`) and again
    at the end (`"completed"`, `"failed"`, or `"interrupted"`).

    `tokenizer_vocab_size` is required and keyword-only (not merely
    required) specifically so it can never be silently misinterpreted as
    a positionally-passed `log_fn` by an existing caller, or vice versa -
    `log_fn` keeps its original positional slot. Pass the tokenizer's real
    vocab size, or `cfg.model.vocab_size` itself if the caller has no
    tokenizer (e.g. synthetic data in tests), which trivially satisfies
    the check.

    A metadata-write failure is never conflated with a training failure:
    if training completes successfully but writing the final
    run_metadata.json then fails, this raises `RunMetadataWriteError`
    (not a training exception) and never returns a misleading status. If
    training itself fails or is interrupted (including Ctrl-C), that
    original exception always propagates unchanged, even if the
    metadata-write attempt that follows it also fails.

    Limitation: none of this can run at all if the process is killed with
    SIGKILL (or otherwise loses the CPU with no chance to run cleanup
    code) - in that case run_metadata.json simply stops being updated,
    permanently reporting whatever status it last recorded. No userspace
    code can guarantee otherwise.

    Returns the final log entry recorded, mainly for test/inspection.
    """
    validate_train_config(cfg)
    validate_vocab_size(cfg, tokenizer_vocab_size)
    verify_training_paths(cfg)

    device = resolve_device(cfg.train.device)
    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.train.seed)

    model = TransformerLM(cfg.model).to(device)
    optimizer = build_optimizer(model, cfg.train.lr, cfg.train.weight_decay)
    device_type, scaler_enabled = resolve_amp_settings(device, cfg.train.amp)
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    Path(cfg.train.log_dir).mkdir(parents=True, exist_ok=True)
    run_id = _new_run_id()
    jsonl_path = Path(cfg.train.log_dir) / f"train_log_{run_id}.jsonl"
    run_metadata_path = Path(cfg.train.log_dir) / f"run_metadata_{run_id}.json"
    run_start_time = time.time()

    start_step = 0
    last_log: dict = {}
    latest_checkpoint_path: Path | None = None

    # step=0 here reflects the pre-resume state; if this run resumes below,
    # the authoritative per-step record lives in the JSONL stream, and the
    # final run_metadata write (see below) reports the true last step.
    run_metadata = build_run_metadata(cfg, step=0)
    run_metadata.update(
        run_id=run_id,
        log_file=jsonl_path.name,
        checkpoint_dir=cfg.train.checkpoint_dir,
        last_checkpoint=None,
        status="running",
    )

    # exclusive=True: this first write is what actually claims run_id -
    # refuse to silently overwrite a stale run_metadata.json left over
    # from an unrelated earlier attempt at the same id (however unlikely,
    # since run_id includes a random component) even if that earlier
    # attempt's JSONL file is absent. Checked before the JSONL file is
    # even created, so neither file is touched on a collision.
    _atomic_write_json(run_metadata_path, run_metadata, exclusive=True)

    # "x" (exclusive create): the same guarantee for the JSONL file - fail
    # loudly rather than silently appending our records onto another
    # run's log.
    with open(jsonl_path, "x") as jsonl_f:

        def log_jsonl(record: dict) -> None:
            jsonl_f.write(_dumps_strict(record) + "\n")
            jsonl_f.flush()

        training_exception: BaseException | None = None
        try:
            if cfg.train.resume:
                latest = find_latest_checkpoint(cfg.train.checkpoint_dir)
                if latest is not None:
                    state = load_checkpoint(
                        latest, model, optimizer, scaler=scaler, map_location=device
                    )
                    start_step = state["step"] + 1
                    latest_checkpoint_path = latest
                    # Surface config-mismatch status explicitly rather than
                    # silently treating any resume as a verified exact one - see
                    # afllm.checkpoint.check_resume_compatibility for what counts
                    # as a critical mismatch vs. a benign environment difference.
                    # Unlike scripts/evaluate_test.py, a training resume never
                    # blocks on a mismatch - it only logs it.
                    compatibility = check_resume_compatibility(state.get("extra"), cfg)
                    resumed_event = {
                        "event": "resumed",
                        "from_checkpoint": str(latest),
                        "step": start_step,
                        "resume_compatibility": compatibility,
                    }
                    log_fn(resumed_event)
                    log_jsonl(resumed_event)

            # Fail fast and clearly if the training set produced zero batches
            # (e.g. too short for the configured seq_len/batch_size), rather than
            # discovering it only once the loop below re-iterates and finds
            # nothing either.
            try:
                train_loader_len = len(train_loader)
            except TypeError:
                train_loader_len = None  # an iterable without a defined length
            if train_loader_len == 0:
                raise EmptyTrainLoaderError(
                    "train_loader produced zero batches - cannot train. Check that "
                    "the training dataset is large enough for the configured "
                    "seq_len and batch_size (see TokenBlockDataset)."
                )

            train_iter = iter(train_loader)

            for step in range(start_step, cfg.train.max_steps):
                lr = cosine_lr(
                    step, cfg.train.warmup_steps, cfg.train.max_steps, cfg.train.lr, cfg.train.min_lr
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr

                optimizer.zero_grad(set_to_none=True)
                t0 = time.time()
                total_loss = 0.0

                for _ in range(cfg.train.grad_accum_steps):
                    try:
                        x, y = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_loader)
                        try:
                            x, y = next(train_iter)
                        except StopIteration:
                            # Re-iterating produced nothing either - the loader
                            # has become empty (or always was, and the upfront
                            # len() check above didn't catch it, e.g. an iterable
                            # without __len__). Fail with a clear error instead of
                            # letting a bare StopIteration escape this function
                            # and propagate as a confusing crash to the caller.
                            raise EmptyTrainLoaderError(
                                "train_loader produced zero batches on "
                                "re-iteration at step "
                                f"{step} - cannot continue training."
                            ) from None
                    x, y = x.to(device), y.to(device)

                    with torch.autocast(
                        device_type=device_type,
                        enabled=cfg.train.amp,
                        dtype=torch.float16 if device_type == "cuda" else torch.bfloat16,
                    ):
                        _, loss = model(x, y)
                        loss = loss / cfg.train.grad_accum_steps
                    scaler.scale(loss).backward()
                    total_loss += loss.item()

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()

                dt = time.time() - t0
                last_log = {
                    "step": step,
                    "train_loss": total_loss,
                    "lr": lr,
                    "sec_per_step": dt,
                }

                if step % cfg.train.eval_interval == 0 or step == cfg.train.max_steps - 1:
                    metrics = evaluate(model, val_loader, device, max_steps=cfg.train.eval_steps)
                    if metrics["valid"]:
                        last_log["val_loss"] = metrics["loss"]
                        last_log["val_perplexity"] = metrics["perplexity"]
                    else:
                        # Never let an invalid eval (e.g. empty validation set)
                        # flow into logs as NaN - record an explicit status instead
                        # of a number that looks real but isn't.
                        last_log["val_status"] = "no_validation_data"

                log_fn(last_log)
                log_jsonl(last_log)

                if step % cfg.train.checkpoint_interval == 0 or step == cfg.train.max_steps - 1:
                    latest_checkpoint_path = save_checkpoint(
                        cfg.train.checkpoint_dir, step, model, optimizer,
                        scaler=scaler,
                        extra=build_run_metadata(cfg, step),
                    )
        except BaseException as exc:  # noqa: BLE001 - deliberately broad, see docstring
            training_exception = exc
    # `jsonl_f` is closed here (the `with` block above always exits
    # cleanly, since any exception was caught inside it), regardless of
    # whether training succeeded, failed, or was interrupted.

    if training_exception is None:
        status = "completed"
    elif isinstance(training_exception, (KeyboardInterrupt, SystemExit)):
        status = "interrupted"
    else:
        status = "failed"

    if "val_loss" in last_log:
        final_val_status = "ok"
    elif last_log.get("val_status") == "no_validation_data":
        final_val_status = "no_validation_data"
    else:
        final_val_status = "not_evaluated"

    run_metadata.update(
        status=status,
        duration_sec=time.time() - run_start_time,
        final_step=last_log.get("step"),
        final_val_status=final_val_status,
        final_val_loss=last_log.get("val_loss"),
        final_val_perplexity=last_log.get("val_perplexity"),
        last_checkpoint=str(latest_checkpoint_path) if latest_checkpoint_path else None,
    )
    if training_exception is not None:
        run_metadata["error"] = f"{type(training_exception).__name__}: {training_exception}"

    try:
        _atomic_write_json(run_metadata_path, run_metadata)
    except OSError as metadata_exc:
        if training_exception is not None:
            # Training already failed/was interrupted - a secondary
            # metadata-write problem must never replace or hide that
            # original outcome. Surface it separately and re-raise the
            # real exception unchanged.
            log_fn({
                "event": "run_metadata_write_failed",
                "error": str(metadata_exc),
                "training_status": status,
            })
            raise training_exception
        # Training itself succeeded. The ONLY problem is persisting the
        # record of that success - this must never be reported as if
        # training had failed.
        raise RunMetadataWriteError(
            f"Training completed successfully through step {last_log.get('step')}, "
            f"but writing {run_metadata_path} failed: {metadata_exc}. Training "
            f"itself did NOT fail - only run-metadata persistence did."
        ) from metadata_exc

    if training_exception is not None:
        raise training_exception
    return last_log
