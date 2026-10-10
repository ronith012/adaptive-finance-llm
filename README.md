# adaptive-finance-llm

A research-oriented, compact language model built from scratch, used as a
testbed for:

- Efficient transformer architectures (rotary embeddings, grouped-query
  attention, and other cheap-to-try variants).
- Financial reasoning and reliable tool use, as the project matures.

This is early-stage research code, not a product. Results reported in
[`docs/experiments/`](docs/experiments/) are preliminary single runs unless
stated otherwise — see the note at the top of each experiment doc.

**The current model architecture (RoPE, RMSNorm, SwiGLU) is a provisional
research baseline and is not equivalent to, or a reproduction of, an
earlier learned-positional-embedding / LayerNorm / GELU-MLP baseline.**
See [`docs/architecture_notes.md`](docs/architecture_notes.md) for the
component-by-component comparison and why these are not interchangeable.

## Status

Baseline PyTorch decoder-only transformer, config-driven training loop,
checkpoint recovery, and a first (preliminary) WikiText-2 smoke test. No
architectural or financial-reasoning claims yet.

## Repository layout

```
configs/                YAML experiment configs
src/afllm/               Library code
  config.py              Dataclass config schema + YAML loader + vocab-size validation
  model.py                Transformer model (RoPE, RMSNorm, GQA-ready) — see docs/architecture_notes.md
  data.py                 Dataset loading/tokenization (WikiText-2 via HF datasets)
  train.py                 Training loop (AMP, grad accum, cosine LR, checkpointing)
  evaluate.py              Validation loop (loss, perplexity)
  checkpoint.py            Save/load/resume checkpoint utilities, incl. RNG state
  env_checks.py            Google Drive mount verification (fail loudly, not silently)
scripts/train.py          CLI entrypoint
scripts/colab_quickstart.ipynb   Colab notebook (mounts Drive, runs training)
tests/                   pytest suite (model, config, checkpoint, env checks — no network needed)
docs/architecture_notes.md   What this model is vs. the original Colab baseline
docs/experiments/        Experiment write-ups, explicitly labeled preliminary/final
```

## Design principles for this repo

- **No large files in git.** Datasets, checkpoints, and logs are never
  committed; they live on Google Drive. See `.gitignore`.
- **Train on Colab, not locally.** The reference environment is a Colab
  Tesla T4. `scripts/train.py` and the config defaults assume dataset cache,
  checkpoints, and logs are under a mounted Drive path.
- **Config-driven, not notebook-driven.** All hyperparameters live in
  `configs/*.yaml`; the notebook just drives `scripts/train.py`.
- **Testable without a GPU or network.** The test suite exercises the model,
  config, and checkpoint logic on tiny synthetic configs/tensors, so it runs
  anywhere, including a laptop with no internet access.

## Setup

See [`SETUP.md`](SETUP.md) for local dev setup (for running tests) and the
Colab + Google Drive training workflow.

## Running tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

These tests do not download any dataset or model weights; they use small
random tensors and tiny model configs.

## Training (Colab)

```bash
python scripts/train.py --config configs/baseline_small.yaml
```

See [`SETUP.md`](SETUP.md) for the full Colab + Drive walkthrough.
