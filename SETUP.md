# Setup

## Local development (tests only — no training, no downloads)

Local machines are for writing/reviewing code and running the test suite.
Do not download datasets or model weights locally; training runs on Colab
(see below).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
```

All tests run on tiny synthetic tensors/configs and require no network access
and no GPU.

## Training on Google Colab (Tesla T4)

1. **Mount Google Drive** and create the project folder structure once:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   import os
   for sub in ["data", "checkpoints", "logs"]:
       os.makedirs(f"/content/drive/MyDrive/adaptive-finance-llm/{sub}", exist_ok=True)
   ```

2. **Clone the repo** (code only — small, fine for the Colab filesystem):

   ```python
   !git clone https://github.com/ronith012/adaptive-finance-llm.git
   %cd adaptive-finance-llm
   !pip install -q -r requirements.txt -r requirements-train.txt  # see note below
   ```

   Training additionally needs `transformers` for the tokenizer (not in the
   base `requirements.txt` since it's only needed for data loading, not for
   the test suite):

   ```python
   !pip install -q transformers
   ```

3. **Run training**, pointing at the WikiText-2 config. Dataset cache,
   checkpoints, and logs all resolve to the Drive paths set in
   `configs/baseline_small.yaml`:

   ```python
   !python scripts/train.py --config configs/baseline_small.yaml
   ```

4. **Resuming after a disconnect**: training auto-resumes from the latest
   checkpoint in `train.checkpoint_dir` when `train.resume: true` (the
   default). Just re-run the same command after remounting Drive.

   Before touching any Drive path, `scripts/train.py` and `train()` both
   call `afllm.env_checks.verify_training_paths`, which checks that
   `/content/drive` is actually mounted. If Drive was not (re)mounted after
   a disconnect, this raises `StorageNotMountedError` with instructions,
   instead of silently creating a look-alike path on ephemeral local disk
   and losing everything on the next recycle.

   **Reproducibility scope on resume:** checkpoints save and restore
   Python, NumPy, and PyTorch (CPU + CUDA) RNG state, so the random stream
   driving dropout etc. continues correctly from a resume. This does
   *not* make a resumed run bit-identical to an uninterrupted one — the
   `DataLoader` iterator restarts from the beginning of a fresh shuffle
   rather than being fast-forwarded to its exact in-epoch position, and
   `num_workers > 0` worker RNG seeding isn't currently pinned to the
   checkpointed state. A resumed run is valid but not bit-reproducible
   against a never-interrupted one.

A ready-to-use version of this workflow is in
[`scripts/colab_quickstart.ipynb`](scripts/colab_quickstart.ipynb).

## What never happens locally

- No dataset downloads (WikiText-2 or otherwise).
- No model weight downloads.
- No training runs — those execute on Colab against Drive-backed storage.

Local work is limited to code, config, tests, and documentation.
