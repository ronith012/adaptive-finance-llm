"""Training loop with grad accumulation, AMP, logging, and checkpoint recovery."""

from __future__ import annotations

import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from afllm.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from afllm.config import ExperimentConfig
from afllm.env_checks import verify_training_paths
from afllm.evaluate import evaluate
from afllm.model import TransformerLM


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


def train(cfg: ExperimentConfig, train_loader, val_loader, log_fn=print) -> dict:
    """Run training. `log_fn` receives a dict per logged step (so callers
    can route logs to a file, Colab cell output, etc).

    Returns the final log entry recorded, mainly for test/inspection.
    """
    verify_training_paths(cfg)

    device = cfg.train.device if torch.cuda.is_available() else "cpu"
    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.train.seed)

    model = TransformerLM(cfg.model).to(device)
    optimizer = build_optimizer(model, cfg.train.lr, cfg.train.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.amp and device == "cuda")

    start_step = 0
    if cfg.train.resume:
        latest = find_latest_checkpoint(cfg.train.checkpoint_dir)
        if latest is not None:
            state = load_checkpoint(latest, model, optimizer, map_location=device)
            start_step = state["step"] + 1
            log_fn({"event": "resumed", "from_checkpoint": str(latest), "step": start_step})

    Path(cfg.train.log_dir).mkdir(parents=True, exist_ok=True)
    last_log: dict = {}
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
                x, y = next(train_iter)
            x, y = x.to(device), y.to(device)

            with torch.autocast(
                device_type="cuda" if device == "cuda" else "cpu",
                enabled=cfg.train.amp,
                dtype=torch.float16 if device == "cuda" else torch.bfloat16,
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
            save_checkpoint(cfg.train.checkpoint_dir, step, model, optimizer)

    return last_log
