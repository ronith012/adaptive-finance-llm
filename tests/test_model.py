import torch

from afllm.config import ModelConfig
from afllm.model import TransformerLM


def tiny_config(**overrides) -> ModelConfig:
    base = dict(
        vocab_size=97,
        dim=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=16,
        dropout=0.0,
    )
    base.update(overrides)
    return ModelConfig(**base)


def test_forward_shapes():
    cfg = tiny_config()
    model = TransformerLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = model(x)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert loss is None


def test_forward_with_targets_computes_loss():
    cfg = tiny_config()
    model = TransformerLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = model(x, y)
    assert loss is not None
    assert loss.item() > 0
    assert torch.isfinite(loss)


def test_backward_pass_updates_params():
    cfg = tiny_config()
    model = TransformerLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))

    before = [p.clone() for p in model.parameters()]
    _, loss = model(x, y)
    loss.backward()

    grads_exist = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert grads_exist

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.step()
    after = list(model.parameters())
    changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert changed


def test_sequence_length_over_max_raises():
    cfg = tiny_config(max_seq_len=4)
    model = TransformerLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    try:
        model(x)
        assert False, "expected assertion error for sequence too long"
    except AssertionError:
        pass


def test_grouped_query_attention_shapes():
    cfg = tiny_config(n_heads=4, n_kv_heads=1)
    model = TransformerLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    logits, _ = model(x)
    assert logits.shape == (1, 8, cfg.vocab_size)


def test_tied_embeddings_share_storage():
    cfg = tiny_config()
    model = TransformerLM(cfg)
    assert model.lm_head.weight.data_ptr() == model.tok_embed.weight.data_ptr()
