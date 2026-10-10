"""Configuration schema and YAML loading for afllm."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int | None = None  # set < n_heads for grouped-query attention
    max_seq_len: int = 512
    dropout: float = 0.0
    rope_theta: float = 10000.0
    tie_embeddings: bool = True


@dataclass
class DataConfig:
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    tokenizer_name: str = "gpt2"
    cache_dir: str = "/content/drive/MyDrive/adaptive-finance-llm/data"
    seq_len: int = 512
    num_workers: int = 2


@dataclass
class TrainConfig:
    batch_size: int = 16
    grad_accum_steps: int = 1
    max_steps: int = 500
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 50
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_steps: int = 50
    checkpoint_interval: int = 100
    checkpoint_dir: str = "/content/drive/MyDrive/adaptive-finance-llm/checkpoints"
    log_dir: str = "/content/drive/MyDrive/adaptive-finance-llm/logs"
    seed: int = 1337
    amp: bool = True  # mixed precision, useful on T4
    device: str = "cuda"
    resume: bool = True  # auto-resume from latest checkpoint if present


@dataclass
class ExperimentConfig:
    name: str = "baseline_small"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _merge_dataclass(instance: Any, overrides: dict) -> Any:
    updates = {}
    for key, value in overrides.items():
        if not hasattr(instance, key):
            raise ValueError(f"Unknown config key '{key}' for {type(instance).__name__}")
        current = getattr(instance, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            updates[key] = _merge_dataclass(current, value)
        else:
            updates[key] = value
    return dataclasses.replace(instance, **updates)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an ExperimentConfig from a YAML file, overriding dataclass defaults."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    cfg = ExperimentConfig()
    return _merge_dataclass(cfg, raw)


def config_to_dict(cfg: ExperimentConfig) -> dict:
    return dataclasses.asdict(cfg)


class VocabSizeMismatchError(ValueError):
    pass


def validate_vocab_size(cfg: ExperimentConfig, tokenizer_vocab_size: int) -> None:
    """Fail loudly if the model's embedding/output size doesn't match the
    tokenizer actually in use.

    A mismatch here produces no error at config-load time — it surfaces
    later as an out-of-range embedding index (or silently truncated output
    logits) once real token ids start flowing through the model. Call this
    right after the tokenizer/dataset is loaded and before training starts.
    """
    if cfg.model.vocab_size != tokenizer_vocab_size:
        raise VocabSizeMismatchError(
            f"model.vocab_size ({cfg.model.vocab_size}) does not match the "
            f"tokenizer's vocab_size ({tokenizer_vocab_size}) for tokenizer "
            f"'{cfg.data.tokenizer_name}'. Update model.vocab_size in your "
            f"config to {tokenizer_vocab_size}, or token ids from this "
            f"tokenizer will index outside the embedding table at runtime."
        )
