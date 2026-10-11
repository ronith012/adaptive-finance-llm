from pathlib import Path

import pytest

from afllm.config import (
    DataConfig,
    ExperimentConfig,
    TrainConfigError,
    VocabSizeMismatchError,
    load_config,
    validate_train_config,
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


def test_validate_train_config_passes_on_default():
    cfg = ExperimentConfig()
    validate_train_config(cfg)  # should not raise


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("grad_accum_steps", 0),
        ("grad_accum_steps", -1),
        ("eval_interval", 0),
        ("eval_interval", -5),
        ("checkpoint_interval", 0),
        ("checkpoint_interval", -5),
        ("max_steps", 0),
        ("max_steps", -1),
    ],
)
def test_validate_train_config_rejects_non_positive_values(field, bad_value):
    cfg = ExperimentConfig()
    setattr(cfg.train, field, bad_value)
    with pytest.raises(TrainConfigError, match=field):
        validate_train_config(cfg)


def test_validate_train_config_reports_all_errors_together():
    cfg = ExperimentConfig()
    cfg.train.grad_accum_steps = 0
    cfg.train.eval_interval = 0
    with pytest.raises(TrainConfigError) as exc_info:
        validate_train_config(cfg)
    message = str(exc_info.value)
    assert "grad_accum_steps" in message
    assert "eval_interval" in message


@pytest.mark.parametrize(
    "field",
    ["grad_accum_steps", "eval_interval", "checkpoint_interval", "max_steps"],
)
@pytest.mark.parametrize("bad_value", [1.5, 2.0, 0.5])
def test_validate_train_config_rejects_fractional_floats(field, bad_value):
    # Regression test: these previously passed a bare `value < 1` check
    # (e.g. 1.5 >= 1) and only failed much later, inside the training
    # loop, as a confusing `TypeError` from `range(1.5)`.
    cfg = ExperimentConfig()
    setattr(cfg.train, field, bad_value)
    with pytest.raises(TrainConfigError, match=field):
        validate_train_config(cfg)


@pytest.mark.parametrize(
    "field",
    ["grad_accum_steps", "eval_interval", "checkpoint_interval", "max_steps"],
)
@pytest.mark.parametrize("bad_value", [True, False])
def test_validate_train_config_rejects_booleans(field, bad_value):
    # bool is technically an int subclass in Python (True == 1, False ==
    # 0), so without an explicit check, True would silently pass a bare
    # `value < 1` check (True == 1 >= 1) - a step count of True/False is
    # never a meaningful config value regardless of what it compares equal
    # to.
    cfg = ExperimentConfig()
    setattr(cfg.train, field, bad_value)
    with pytest.raises(TrainConfigError, match=field):
        validate_train_config(cfg)


def test_validate_train_config_still_accepts_plain_positive_ints():
    # Non-regression: ordinary int values (the normal case) must still
    # pass after the stricter type checking was added.
    cfg = ExperimentConfig()
    cfg.train.grad_accum_steps = 4
    cfg.train.eval_interval = 10
    cfg.train.checkpoint_interval = 10
    cfg.train.max_steps = 1000
    validate_train_config(cfg)  # should not raise
