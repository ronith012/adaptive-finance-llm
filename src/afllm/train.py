"""Training loop with grad accumulation, AMP, logging, and checkpoint recovery."""

from __future__ import annotations

import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from afllm.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from afllm.config import ExperimentConfig, validate_train_config
from afllm.env_checks import resolve_device, verify_training_paths
from afllm.evaluate import evaluate
from afllm.model import TransformerLM


class EmptyTrainLoaderError(RuntimeError):
    pass


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


def train(cfg: ExperimentConfig, train_loader, val_loader, log_fn=print) -> dict:
    """Run training. `log_fn` receives a dict per logged step (so callers
    can route logs to a file, Colab cell output, etc).

    Returns the final log entry recorded, mainly for test/inspection.
    """
    validate_train_config(cfg)
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

    start_step = 0
    if cfg.train.resume:
        latest = find_latest_checkpoint(cfg.train.checkpoint_dir)
        if latest is not None:
            state = load_checkpoint(latest, model, optimizer, scaler=scaler, map_location=device)
            start_step = state["step"] + 1
            log_fn({"event": "resumed", "from_checkpoint": str(latest), "step": start_step})

    Path(cfg.train.log_dir).mkdir(parents=True, exist_ok=True)
    last_log: dict = {}

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
            last_log["val_loss"] = metrics["loss"]
            last_log["val_perplexity"] = metrics["perplexity"]

        log_fn(last_log)

        if step % cfg.train.checkpoint_interval == 0 or step == cfg.train.max_steps - 1:
            save_checkpoint(cfg.train.checkpoint_dir, step, model, optimizer, scaler=scaler)

    return last_log
