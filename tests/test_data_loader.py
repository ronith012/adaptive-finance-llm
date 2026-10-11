"""Tests for the WikiText-2 dataset loader.

`afllm.data.load_wikitext2` lazily imports `datasets` and `transformers`
(neither is a hard dependency of this package's core - see
afllm/data.py's module docstring). These tests fake both out via
`sys.modules` injection so they exercise the real `load_wikitext2` code
path - including exactly which repo id, dataset config, and cache_dir it
passes through - with zero network access and no real dependency on
`datasets`/`transformers` being installed.
"""

import sys
import types

from afllm.config import DataConfig
from afllm.data import load_wikitext2


class _FakeTokenizer:
    pad_token = None
    eos_token = "<eos>"
    vocab_size = 123

    def __call__(self, text, return_tensors=None):
        # One token id per word, enough to exercise encode_split's shape.
        return {"input_ids": [1] * len(text.split())}


def _install_fake_datasets_and_transformers(monkeypatch, captured):
    def fake_load_dataset(name, config, cache_dir=None):
        captured["dataset_name"] = name
        captured["dataset_config"] = config
        captured["cache_dir"] = cache_dir
        return {
            "train": {"text": ["hello world"]},
            "validation": {"text": ["val text here"]},
            "test": {"text": ["test text"]},
        }

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    fake_tokenizer_instance = _FakeTokenizer()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name):
            captured["tokenizer_name"] = name
            return fake_tokenizer_instance

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def test_load_wikitext2_uses_configured_salesforce_repo_id(monkeypatch):
    captured = {}
    _install_fake_datasets_and_transformers(monkeypatch, captured)

    cfg = DataConfig(cache_dir="/tmp/afllm-test-cache")
    corpus = load_wikitext2(cfg)

    assert captured["dataset_name"] == "Salesforce/wikitext"
    assert captured["dataset_config"] == "wikitext-2-raw-v1"
    assert captured["cache_dir"] == "/tmp/afllm-test-cache"
    assert corpus.vocab_size == 123


def test_load_wikitext2_cache_dir_is_a_genuine_pass_through_not_hardcoded(monkeypatch):
    """Use a cache_dir that looks nothing like the project default, to
    prove it's forwarded verbatim rather than a hardcoded literal inside
    load_wikitext2 happening to match the common default."""
    captured = {}
    _install_fake_datasets_and_transformers(monkeypatch, captured)

    cfg = DataConfig(cache_dir="/completely/different/location/xyz")
    load_wikitext2(cfg)

    assert captured["cache_dir"] == "/completely/different/location/xyz"


def test_load_wikitext2_respects_dataset_name_override(monkeypatch):
    """A config pointing at a different dataset_name must reach
    load_dataset unchanged - dataset_name is a real parameter, not a
    literal baked into load_wikitext2."""
    captured = {}
    _install_fake_datasets_and_transformers(monkeypatch, captured)

    cfg = DataConfig(dataset_name="some-other/dataset", cache_dir="/tmp/x")
    load_wikitext2(cfg)

    assert captured["dataset_name"] == "some-other/dataset"


def test_load_wikitext2_respects_tokenizer_name(monkeypatch):
    captured = {}
    _install_fake_datasets_and_transformers(monkeypatch, captured)

    cfg = DataConfig(tokenizer_name="gpt2", cache_dir="/tmp/x")
    load_wikitext2(cfg)

    assert captured["tokenizer_name"] == "gpt2"


def test_load_wikitext2_returns_all_three_splits(monkeypatch):
    captured = {}
    _install_fake_datasets_and_transformers(monkeypatch, captured)

    cfg = DataConfig(cache_dir="/tmp/x")
    corpus = load_wikitext2(cfg)

    assert corpus.train.numel() > 0
    assert corpus.validation.numel() > 0
    assert corpus.test.numel() > 0
