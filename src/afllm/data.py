"""Dataset loading and tokenization.

Downloads/caches go to `DataConfig.cache_dir`, which in the Colab
workflow is a path under a mounted Google Drive — never local disk
on a dev machine. This module is import-safe without network access;
network calls only happen inside `load_wikitext2`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from afllm.config import DataConfig


class TokenBlockDataset(Dataset):
    """Wraps a flat 1D tensor of token ids into fixed-length blocks."""

    def __init__(self, token_ids: torch.Tensor, seq_len: int):
        self.token_ids = token_ids
        self.seq_len = seq_len
        self.n_blocks = (len(token_ids) - 1) // seq_len

    def __len__(self) -> int:
        return max(self.n_blocks, 0)

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        end = start + self.seq_len
        x = self.token_ids[start:end]
        y = self.token_ids[start + 1 : end + 1]
        return x, y


@dataclass
class TokenizedCorpus:
    train: torch.Tensor
    validation: torch.Tensor
    test: torch.Tensor
    vocab_size: int


def load_wikitext2(cfg: DataConfig) -> TokenizedCorpus:
    """Load and tokenize WikiText-2. Requires `datasets` + `tokenizers`/HF hub.

    This performs network I/O and should only be run in the training
    environment (Colab), with `cfg.cache_dir` pointed at Google Drive.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, cache_dir=cfg.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def encode_split(split: str) -> torch.Tensor:
        text = "\n\n".join(ds[split]["text"])
        ids = tokenizer(text, return_tensors=None)["input_ids"]
        return torch.tensor(ids, dtype=torch.long)

    return TokenizedCorpus(
        train=encode_split("train"),
        validation=encode_split("validation"),
        test=encode_split("test"),
        vocab_size=tokenizer.vocab_size,
    )


def make_loaders(corpus: TokenizedCorpus, cfg: DataConfig, batch_size: int):
    from torch.utils.data import DataLoader

    train_ds = TokenBlockDataset(corpus.train, cfg.seq_len)
    val_ds = TokenBlockDataset(corpus.validation, cfg.seq_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=cfg.num_workers, drop_last=True,
    )
    return train_loader, val_loader
