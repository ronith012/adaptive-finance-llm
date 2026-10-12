import torch

from afllm.config import DataConfig
from afllm.data import TokenizedCorpus, make_test_loader


def _fake_corpus(test_len: int, vocab_size: int = 32) -> TokenizedCorpus:
    return TokenizedCorpus(
        train=torch.empty(0, dtype=torch.long),
        validation=torch.empty(0, dtype=torch.long),
        test=torch.arange(test_len),
        vocab_size=vocab_size,
    )


def test_make_test_loader_yields_fixed_length_blocks():
    corpus = _fake_corpus(test_len=100)
    cfg = DataConfig(seq_len=8, num_workers=0)

    loader = make_test_loader(corpus, cfg, batch_size=4)
    batches = list(loader)

    assert len(batches) > 0
    for x, y in batches:
        assert x.shape == y.shape
        assert x.shape[-1] == cfg.seq_len


def test_make_test_loader_is_unshuffled_and_covers_blocks_in_order():
    corpus = _fake_corpus(test_len=100)
    cfg = DataConfig(seq_len=8, num_workers=0)

    loader = make_test_loader(corpus, cfg, batch_size=1)
    first_x, _ = next(iter(loader))

    # Unshuffled: the very first block must start at token 0.
    assert torch.equal(first_x[0], torch.arange(0, 8))


def test_make_test_loader_drops_incomplete_trailing_batch():
    # test_len=100, seq_len=8 -> n_blocks = (100-1)//8 = 12. batch_size=5 ->
    # 12//5 = 2 full batches, with drop_last=True dropping the 2-block remainder.
    corpus = _fake_corpus(test_len=100)
    cfg = DataConfig(seq_len=8, num_workers=0)

    loader = make_test_loader(corpus, cfg, batch_size=5)
    batches = list(loader)

    assert len(batches) == 2
    for x, _ in batches:
        assert x.shape[0] == 5
