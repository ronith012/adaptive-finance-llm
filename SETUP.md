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

A ready-to-use version of everything below is in
[`scripts/colab_quickstart.ipynb`](scripts/colab_quickstart.ipynb) — open it
in Colab and run the cells top to bottom. It requires no local Mac
environment; the clone, install, smoke test, and training all happen inside
the Colab session itself.

1. **Mount Google Drive** and create the project folder structure once:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   import os
   assert os.path.ismount('/content/drive'), "Drive did not mount — re-run and approve the prompt."

   for sub in ["data", "checkpoints", "logs", "smoke_test"]:
       os.makedirs(f"/content/drive/MyDrive/adaptive-finance-llm/{sub}", exist_ok=True)
   ```

2. **Clone the repo**, pointing at the branch under review (not necessarily
   `main` — check `scripts/colab_quickstart.ipynb`'s `REPO_BRANCH` variable
   or the PR you're validating):

   ```python
   !git clone -b feature/initial-scaffold --single-branch \
       https://github.com/ronith012/adaptive-finance-llm.git
   %cd adaptive-finance-llm
   ```

3. **Install dependencies** — deliberately *not* `pip install -r
   requirements.txt`. Colab's preinstalled `torch` is already built against
   this runtime's CUDA version; blindly installing `requirements.txt`'s
   `torch>=2.1` risks pip swapping in a CPU-only or mismatched-CUDA wheel.
   `pytest` is dev/test-only and isn't needed to run training or the smoke
   test, so it's skipped here too:

   ```python
   !pip install -q numpy pyyaml datasets tokenizers tqdm transformers
   ```

   Then confirm the GPU is actually visible to torch before doing anything
   else (catches both a missing GPU runtime and an accidentally-reinstalled
   CPU-only torch):

   ```python
   import torch
   assert torch.cuda.is_available(), "CUDA not available — check Runtime > Change runtime type."
   ```

4. **Run the smoke test first.** This builds a tiny model, does one
   forward/backward pass, and saves+reloads a checkpoint — no dataset
   download, finishes in seconds, and (if pointed at Drive) also confirms
   Drive write access before you commit to a long run:

   ```python
   !python scripts/smoke_test.py --device cuda \
       --checkpoint-dir "/content/drive/MyDrive/adaptive-finance-llm/smoke_test"
   ```

   Expect to see `[1/5]` through `[5/5]` print `OK`/`SMOKE TEST PASSED`. If
   this fails, fix the environment before running real training — a failure
   here (e.g. a CUDA mismatch or a Drive permissions issue) will also sink
   a full training run, just after wasting much more time first.

   You can also run it locally on this Mac with no GPU and no Drive, purely
   to sanity-check the code before ever opening Colab:

   ```bash
   python scripts/smoke_test.py --device cpu
   ```

5. **Run training**, pointing at the WikiText-2 config, only after the
   smoke test passes. Dataset cache, checkpoints, and logs all resolve to
   the Drive paths set in `configs/baseline_small.yaml`:

   ```python
   !python scripts/train.py --config configs/baseline_small.yaml
   ```

6. **Resuming after a disconnect**: training auto-resumes from the latest
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
   checkpointed state. **This remains true after the reproducibility work
   below — it has not been changed or fixed, only documented more
   precisely.** A resumed run is valid but not bit-reproducible against a
   never-interrupted one. Do not claim exact continuation; nothing in this
   codebase currently restores exact data order across a resume.

   **Config-mismatch detection on resume:** every checkpoint now records
   the full config, seed, and installed library versions that produced it
   (`afllm.checkpoint.build_run_metadata`). On resume, `train()` compares
   the checkpoint's saved config against the one you're resuming with
   (`afllm.checkpoint.check_resume_compatibility`) for the fields that
   actually affect reproducibility — model architecture, dataset/tokenizer
   identity, seed, LR schedule, batching — and logs one of `"match"`,
   `"mismatch"` (with the specific fields that differ), or `"no_metadata"`
   (a checkpoint saved before this feature existed). This is a visibility
   tool, not a guardrail: it warns, it does not block a resume, and
   operational fields like `checkpoint_dir`/`log_dir` are deliberately
   excluded since they legitimately differ across environments.

## Recommended pinned environment

`requirements.txt` intentionally uses `>=` lower bounds, not exact pins, so
nothing here is pinned yet. The only environment actually validated end to
end on real GPU hardware so far: **PyTorch 2.11.0+cu130 on a Tesla T4**
(the Colab run that validated the smoke test and the RNG-restoration fix).
`transformers`/`datasets`/`tokenizers` versions for that run were not
captured at the time — they were installed unpinned — so they cannot be
honestly reported here.

Going forward, every checkpoint's `extra["environment"]` dict (via
`afllm.checkpoint.capture_environment_versions`) now records the installed
`torch`, `cuda`, `transformers`, `datasets`, and `tokenizers` versions for
the run that produced it. The next time a full validated run completes,
pull that dict from its checkpoint and turn it into an exact-pinned
`requirements-validated.txt` (or equivalent) — don't guess at version
numbers, read them from an actual validated run's own metadata.

## What never happens locally

- No dataset downloads (WikiText-2 or otherwise).
- No model weight downloads.
- No training runs — those execute on Colab against Drive-backed storage.

Local work is limited to code, config, tests, and documentation.
