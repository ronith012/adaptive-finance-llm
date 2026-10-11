import torch

from afllm.data import TokenBlockDataset


def test_input_target_alignment_first_block():
    ids = torch.arange(20)
    ds = TokenBlockDataset(ids, seq_len=5)

    x0, y0 = ds[0]
    assert torch.equal(x0, torch.arange(0, 5))
    assert torch.equal(y0, torch.arange(1, 6))
    assert torch.equal(y0, x0 + 1)


def test_input_target_alignment_middle_block():
    ids = torch.arange(20)
    ds = TokenBlockDataset(ids, seq_len=5)

    x1, y1 = ds[1]
    assert torch.equal(x1, torch.arange(5, 10))
    assert torch.equal(y1, torch.arange(6, 11))


def test_len_matches_expected_block_count():
    # n_blocks = (len(ids) - 1) // seq_len. With len=20, seq_len=5:
    # (20-1)//5 = 3, not 4 - the 4th block's target would need token index
    # 20, which is out of range for a 20-length tensor (valid indices 0..19).
    ids = torch.arange(20)
    ds = TokenBlockDataset(ids, seq_len=5)
    assert len(ds) == 3


def test_last_block_never_reads_out_of_range():
    # len=20, seq_len=5 -> n_blocks = (20-1)//5 = 3, so blocks cover tokens
    # 0..15 only; tokens 16-19 are an incomplete trailing remainder and are
    # intentionally dropped (no partial/padded block is ever produced) -
    # this is expected block-chunking behavior, not a bug.
    ids = torch.arange(20)
    ds = TokenBlockDataset(ids, seq_len=5)
    n = len(ds)
    assert n == 3

    x_last, y_last = ds[n - 1]
    assert len(x_last) == 5
    assert len(y_last) == 5
    assert x_last[-1].item() == 14
    assert y_last[-1].item() == 15  # last token actually used by any block


def test_boundary_sweep_never_returns_short_blocks():
    for total_len in (1, 2, 5, 9, 10, 11, 17, 100):
        for seq_len in (4, 8):
            ids = torch.arange(total_len)
            ds = TokenBlockDataset(ids, seq_len=seq_len)
            for idx in range(len(ds)):
                x, y = ds[idx]
                assert len(x) == seq_len, (total_len, seq_len, idx)
                assert len(y) == seq_len, (total_len, seq_len, idx)
                assert torch.equal(y, x + 1)


def test_dataset_too_short_for_one_block_has_zero_length():
    ids = torch.arange(3)
    ds = TokenBlockDataset(ids, seq_len=8)
    assert len(ds) == 0
