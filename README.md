# HE-NAC - High Fidelity Neural Audio Coding

Training and experiment code forked from [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec) (RVQGAN-style neural codec). Original paper: [High-Fidelity Audio Compression with Improved RVQGAN](https://arxiv.org/abs/2306.06546).

---

## Branches and training stages

The codec is trained in **three stages**. Each stage used to be a separate checkout; in git they are kept as **branches** so you can `git checkout <branch>` to reproduce an old run or `git diff main..<branch>` while consolidating.

| Git branch | Stage (informal) | Role |
|------------|------------------|------|
| **`baseline_skipae`** | Stage 1 — baseline | Train the **core** codec (encoder, main residual VQ, main decoder). Skip modules may exist in the graph for layout reasons but **are not used** in `encode` / forward the same way as later stages. **Full-band** loss target. |
| **`entropy_ctrl`** | Stage 2 — entropy / mid-band | **Resume** from stage 1: load checkpoint with **all `skip_aes.*` keys removed**, then train the **mid-band** skip (`skip_aes.0`) and its band decoder. Loss targets **mid+** bands (e.g. `SplitBands` from index 1). Core weights start from stage 1. |
| **`entropy_ctrl_hb`** | Stage 3 — high-band | **Full** model: **two** skip AEs (`skip_aes.0` / `skip_aes.1`), **two** `multidecoders`, cascade decode (core blind → MB → HB). **Resume** from stage 2: drop only **`skip_aes.1.*`** from the checkpoint so HB starts fresh; loss targets **high** bands (e.g. from index 2). |

**`main`** is where **ongoing development** should land: unified configs, shared `DAC`, and helpers such as `dac/training_stage.py`. Until consolidation is finished, treat the three branches above as **frozen references** for “what the code looked like when that stage was trained.”

---

## Layout (on `main`)

- **`dac/model/dac.py`** — Main `DAC`: encoder, main quantizer, main decoder, `skip_aes` (×2 on the HB-era layout), `multidecoders` (×2), `encode` / `forward` / `infer_bands`.
- **`dac/model/dac_skip.py`** — Skip-path autoencoder (`DACSkip`) per band.
- **`dac/training_stage.py`** — Stage → **which parameters are trainable** (`apply_stage_requires_grad`). Wire this from `scripts/train.py` + YAML when you merge behaviour onto `main`.
- **`scripts/train.py`** — Training loop (resume filters, band slicing, discriminator, losses). Today stage-specific logic still lives here; align it with `training_stage.py` on `main`.
- **`conf/`** — ArgBind YAML; **`conf/final/`** for end-to-end recipes; **`conf/base.yml`** for shared defaults.
- **`resave_ckpt.py`** — Checkpoint reshape / resave between stages (legacy multi-branch workflow).
- **`tests/test_training_stage_grads_32khz.py`** — Smoke test: HB-shaped 32 kHz `DAC`, stage masks, one backward pass; asserts **no grad on frozen** weights and **some** grad on trainable weights (not every RVQ / decoder block gets a grad on a toy scalar loss).

Encode/decode CLI and Docker from upstream DAC still exist but are not the focus of this README.

---

## Configs and training (32 kHz)

Recipes live under **`conf/final/`**. Examples:

- **`32khz_mb.yml`** — Mid-band / stage-2 style settings. Ensure `DAC.skip_args` matches what **`DAC.__init__`** expects on your branch (flat dict vs `{0: …, 1: …}` for two-skip models).
- **`32khz_hb.yml`** — Full HB model: `DAC.skip_args` with **`0:`** and **`1:`** blocks; `resume_ckpt` should point at a finished **stage-2** run; see `decoder_frozen` and trainable prefixes in `scripts/train.py`.

Example:

```bash
export CUDA_VISIBLE_DEVICES=0
python scripts/train.py --args.load conf/final/32khz_hb.yml --save_path runs/my_experiment/
```

Multi-GPU: same pattern as upstream (`torchrun … scripts/train.py …`). Set **`resume_ckpt`**, dataset paths, and lambdas in YAML to your environment.

---

## Install

```bash
pip install -e .
# optional: pip install -e ".[dev]"
```

---

## Tests

```bash
python -m pytest tests/test_training_stage_grads_32khz.py -v
```

Other tests under `tests/` may assume local datasets or assets; run selectively if paths are missing.

---

## Working with branches in practice

1. **Daily work** — Branch off **`main`**, open PRs back to `main`.
2. **Reproduce an old stage** — `git fetch && git checkout baseline_skipae` (or `entropy_ctrl` / `entropy_ctrl_hb`), install, run the same `scripts/train.py` + YAML that was used then.
3. **Port a fix** — Cherry-pick or manually apply from a stage branch onto `main`, then run configs + tests on `main`.
4. **Retire branches** — After `main` matches each stage under config flags, you can delete the old branches or leave them tagged for paper reproducibility.

---

## Consolidation checklist (for `main`)

1. **Single codebase** — One `DAC` + YAML-selected stage (forward parity: “match today’s math” per stage first).
2. **Wire `training_stage.py`** — `training_stage` (or similar) in YAML + `apply_stage_requires_grad` in `scripts/train.py`.
3. **`DAC.skip_args`** — One consistent schema (e.g. `{0: …, 1: …}`) wherever the two-skip model is built.

---

## License

Upstream DAC materials remain under the **MIT** license in `LICENSE` unless replaced by your organization.
