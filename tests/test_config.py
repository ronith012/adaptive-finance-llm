import pytest

from afllm.config import (
    ExperimentConfig,
    VocabSizeMismatchError,
    load_config,
    validate_vocab_size,
)


def test_default_config_constructs():
    cfg = ExperimentConfig()
    assert cfg.model.dim == 256
    assert cfg.train.max_steps == 500


def test_load_config_overrides(tmp_path):
    yaml_content = """
model:
  dim: 64
  n_layers: 2
train:
  max_steps: 10
"""
    path = tmp_path / "test_config.yaml"
    path.write_text(yaml_content)

    cfg = load_config(path)
    assert cfg.model.dim == 64
    assert cfg.model.n_layers == 2
    assert cfg.train.max_steps == 10
    # untouched fields keep defaults
    assert cfg.model.n_heads == 8


def test_load_config_rejects_unknown_key(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("model:\n  not_a_real_field: 1\n")
    try:
        load_config(path)
        assert False, "expected ValueError for unknown config key"
    except ValueError:
        pass


def test_validate_vocab_size_passes_on_match():
    cfg = ExperimentConfig()
    cfg.model.vocab_size = 50257
    validate_vocab_size(cfg, tokenizer_vocab_size=50257)  # should not raise


def test_validate_vocab_size_raises_on_mismatch():
    cfg = ExperimentConfig()
    cfg.model.vocab_size = 32000
    with pytest.raises(VocabSizeMismatchError):
        validate_vocab_size(cfg, tokenizer_vocab_size=50257)
