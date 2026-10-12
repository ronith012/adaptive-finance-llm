"""Tests for scripts/evaluate_test.py.

Fakes `datasets`/`transformers` via sys.modules injection (same technique
as tests/test_data_loader.py) so the success/mismatch paths exercise the
real script logic with zero network access and no real dependency on
those packages being installed.
"""

import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_test  # noqa: E402

from afllm.checkpoint import CheckpointConfigMismatchError, build_run_metadata, save_checkpoint  # noqa: E402
from afllm.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig  # noqa: E402
from afllm.model import TransformerLM  # noqa: E402

VOCAB_SIZE = 37


def _tiny_eval_cfg(tmp_path) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=VOCAB_SIZE, dim=8, n_layers=1, n_heads=2, n_kv_heads=2, max_seq_len=8,
        ),
        data=DataConfig(
            dataset_name="fake/dataset", dataset_config="fake-config",
            tokenizer_name="fake-tokenizer", cache_dir=str(tmp_path / "cache"),
            seq_len=1, num_workers=0,
        ),
        train=TrainConfig(
            device="cpu", checkpoint_dir=str(tmp_path / "checkpoints"),
            log_dir=str(tmp_path / "logs"), batch_size=1, amp=False, resume=False,
        ),
    )


def _write_config_yaml(tmp_path, cfg: ExperimentConfig) -> Path:
    import yaml

    from afllm.config import config_to_dict

    path = tmp_path / "eval_config.yaml"
    path.write_text(yaml.safe_dump(config_to_dict(cfg)))
    return path


class _FakeTokenizer:
    pad_token = None
    eos_token = "<eos>"
    vocab_size = VOCAB_SIZE

    def __call__(self, text, return_tensors=None):
        return {"input_ids": [1] * len(text.split())}


def _install_fake_datasets_and_transformers(monkeypatch):
    def fake_load_dataset(name, config, cache_dir=None):
        return {
            "train": {"text": ["one two"]},
            "validation": {"text": ["one two"]},
            "test": {"text": ["one two"]},
        }

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name):
            return _FakeTokenizer()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def test_evaluate_test_raises_before_touching_dataset_when_no_checkpoint(tmp_path, monkeypatch):
    """No fake datasets/transformers installed here on purpose: if the
    script tried to load the dataset before checking for a checkpoint, it
    would fail with an import/network error instead of the intended
    FileNotFoundError - this test would catch that regression."""
    cfg = _tiny_eval_cfg(tmp_path)
    config_path = _write_config_yaml(tmp_path, cfg)
    monkeypatch.setattr(sys, "argv", ["evaluate_test.py", "--config", str(config_path)])

    with pytest.raises(FileNotFoundError):
        evaluate_test.main()


def test_evaluate_test_success_reports_loss_and_checkpoint_info(tmp_path, monkeypatch, capsys):
    _install_fake_datasets_and_transformers(monkeypatch)
    cfg = _tiny_eval_cfg(tmp_path)
    config_path = _write_config_yaml(tmp_path, cfg)

    model = TransformerLM(cfg.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_path = save_checkpoint(
        cfg.train.checkpoint_dir, step=7, model=model, optimizer=optimizer,
        extra=build_run_metadata(cfg, step=7),
    )

    monkeypatch.setattr(sys, "argv", ["evaluate_test.py", "--config", str(config_path)])
    evaluate_test.main()

    import json

    output = json.loads(capsys.readouterr().out)
    assert output["valid"] is True
    assert isinstance(output["loss"], float)
    assert output["checkpoint"] == str(ckpt_path)
    assert output["checkpoint_step"] == 7


def test_evaluate_test_rejects_incompatible_checkpoint_before_evaluating(tmp_path, monkeypatch):
    _install_fake_datasets_and_transformers(monkeypatch)
    cfg = _tiny_eval_cfg(tmp_path)

    saved_cfg = _tiny_eval_cfg(tmp_path)
    saved_cfg.model.dim = 16  # critically different from cfg.model.dim below
    model = TransformerLM(saved_cfg.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(
        cfg.train.checkpoint_dir, step=0, model=model, optimizer=optimizer,
        extra=build_run_metadata(saved_cfg, step=0),
    )

    config_path = _write_config_yaml(tmp_path, cfg)
    monkeypatch.setattr(sys, "argv", ["evaluate_test.py", "--config", str(config_path)])

    with pytest.raises(CheckpointConfigMismatchError, match="model.dim"):
        evaluate_test.main()
