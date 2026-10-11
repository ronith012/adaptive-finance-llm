from pathlib import Path

import pytest

from afllm.config import (
    DataConfig,
    ExperimentConfig,
    VocabSizeMismatchError,
    load_config,
    validate_vocab_size,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_config_constructs():
    cfg = ExperimentConfig()
    assert cfg.model.dim == 256
    assert cfg.train.max_steps == 500


def test_default_dataset_uses_namespaced_salesforce_repo_id():
    # The bare "wikitext" repo id relied on a script-based HF `datasets`
    # loader that was deprecated/removed on the Hub; "Salesforce/wikitext"
    # is the namespaced, data-only replacement repo for the same dataset.
    cfg = DataConfig()
    assert cfg.dataset_name == "Salesforce/wikitext"
    assert cfg.dataset_config == "wikitext-2-raw-v1"  # unchanged, preserved


def test_cache_dir_remains_configurable_not_hardcoded():
    cfg = DataConfig(cache_dir="/some/custom/path")
    assert cfg.cache_dir == "/some/custom/path"


def test_baseline_small_yaml_uses_salesforce_wikitext_repo_id():
    cfg = load_config(REPO_ROOT / "configs" / "baseline_small.yaml")
    assert cfg.data.dataset_name == "Salesforce/wikitext"
    assert cfg.data.dataset_config == "wikitext-2-raw-v1"


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
