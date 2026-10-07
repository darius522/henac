# HE-NAC

**HE-NAC: High-Efficiency Neural Audio Coding via Hierarchical and Residual
Multi-Band Quantization** is a 32 kHz neural audio codec trained in three
stages. It starts with a core path, freezes it while training a mid band path,
then freezes both while training a high band path. The model shares an encoder
across paths and uses residual vector quantizers at different time resolutions.
At inference, the decoder paths are filtered by frequency and summed.

This repository contains the model, staged training, WAV reconstruction, and
objective evaluation. It builds on the [Descript Audio Codec
(DAC)](https://github.com/descriptinc/descript-audio-codec) implementation.

| Stage | Trained path | Loss target | Initialize from |
| --- | --- | --- | --- |
| Core | Encoder, core RVQ, blind decoder | Full band | Scratch |
| Mid | Mid skip RVQ and decoder | Above 3 kHz | Core checkpoint |
| High | High skip RVQ and decoder | Above 6 kHz | Mid checkpoint |

The stage transition copies the trained decoder tail into the next path,
matching the original branch based training code. Earlier paths are frozen.
Training uses the empirical code entropy for monitoring only: there is no
explicit entropy loss and no entropy coded bitstream.

## Install

Use Python 3.10 or newer and install PyTorch for your CPU or CUDA platform,
then install this package from the repository root:

```bash
pip install -e .
```

The model weights will be distributed separately from Git. **Checkpoint download:
TODO (add the public URL before announcing the release).** The prepared
self-contained 300k checkpoint has SHA-256
`978c7e9950b787b7a3673c198678c711cd5fab2fe766f4e41516125fc35a305e`.
Place it at `checkpoints/henac-300k.pth` and use that path for `--checkpoint`.
The supplied local legacy checkpoint also works as `checkpoints/300k` when
`checkpoints/conf.yaml` is present.
**Checkpoint license: TODO (confirm distribution terms before publishing the
weights).**

## Reconstruct audio

```bash
henac-infer --checkpoint checkpoints/henac-300k.pth \
  --input path/to/audio.wav --output outputs \
  --codebooks 16 4 2
```

`--input` accepts a WAV or FLAC file or a directory of files. Outputs are
32 kHz PCM WAVs, with the input directory structure preserved. The model is
mono; multichannel inputs are averaged and the reconstructed channel is copied
back to the original number of channels. Files longer than ten seconds are
processed with overlapping context. Use `--chunk-seconds` and
`--context-seconds` to change that behavior.

The command also writes `entropy.csv`. Its kbps values estimate the Shannon
entropy of the observed code indices. They are **estimates**, not encoded
file sizes. A core path is required; a high path also requires a mid path.

## Train

Provide separate WAV or FLAC directories, individual files, or CSV manifests
with a `path` column. The examples below run on one GPU from the repository
root:

```bash
CUDA_VISIBLE_DEVICES=0 henac-train --args.load conf/henac/core.yml \
  --train_path /data/train --val_path /data/valid --save_path runs/core

CUDA_VISIBLE_DEVICES=0 henac-train --args.load conf/henac/mid.yml \
  --init_from runs/core/best.pth \
  --train_path /data/train --val_path /data/valid --save_path runs/mid

CUDA_VISIBLE_DEVICES=0 henac-train --args.load conf/henac/high.yml \
  --init_from runs/mid/best.pth \
  --train_path /data/train --val_path /data/valid --save_path runs/high
```

The recipes in `conf/henac/` are supplied in the source checkout and record
the reference architecture and loss settings. A validation checkpoint is
written to `latest.pth`; `best.pth` contains model weights only. To continue
an interrupted run, pass `--resume_from runs/<stage>/latest.pth` with the same
stage recipe. Resume restores the optimizer, scheduler, and data offset, but
does not restore random number generator state, so it is not a bit-exact
continuation.

`--num_iters` is the total desired optimizer step count, including completed
steps. `--init_from` starts a new stage with a fresh optimizer and
discriminator.

For distributed training, run the script with `torchrun`, for example:

```bash
torchrun --nproc_per_node=8 -m henac.train \
  --args.load conf/henac/core.yml \
  --train_path /data/train --val_path /data/valid --save_path runs/core
```

The original training data is not included. The core recipe starts with 32
codebooks and quantizer dropout; the mid and high recipes use 16 core
codebooks. You can supply multiple balanced source groups through
`train/build_dataset.folders` and `val/build_dataset.folders` in a recipe.
The paper reports 300k training steps with an effective batch size of 32. The
historical high-band configuration saved with the local 300k checkpoint has a
400k-step schedule and batch size 8 per process. To stop these recipes at 300k
steps, pass `--num_iters 300000`; four processes with batch size 8 give an
effective batch size of 32. The original training run's process count and
dataset are unavailable, so these recipes are a checked starting point rather
than a claim of an exact full-run reproduction.

## Evaluate

```bash
henac-eval --reference path/to/input --reconstruction outputs
```

The evaluator requires matching relative filenames. It writes per file L1,
SNR, and SI-SDR for the full signal and three frequency bands to
`evaluation.csv`, plus mean values to `evaluation.json`.

## Reference parity

The supplied local 300k checkpoint and 39 MUSHRA pairs were checked against
the original `entropy_ctrl_hb` branch before consolidation. The consolidated
model strictly loads all weights and regenerates **byte identical WAVs** to
that branch on the tested A100 environment. Against the supplied gold WAVs,
the lowest exact PCM fraction is 98.870% and the lowest signal to difference
ratio is 47.96 dB. See [the parity record](docs/reference-parity.md) for
hashes, commands, and the limits of this result.

The CPU tests run without the private reference assets:

```bash
python -m pytest tests -q
ruff check henac scripts tests
ruff format --check henac scripts tests
```

One step training calculations were also compared against the three original
branches, including a check with the supplied final checkpoint. See
[the training parity record](docs/training-parity.md) for results and limits.

## Paper and scope

By Darius Pétermann, Wootaek Lim, Seungkwon Beack, and Minje Kim. See the
[project page](https://minjekim.com/research-projects/he-nac) and
[citation metadata](CITATION.cff). **Paper DOI: TODO (add the published DOI).**
The paper reports effective bitrates after Huffman coding. This release
currently reports empirical Shannon entropy estimates and does not reproduce
those encoded file sizes.

HENAC currently reconstructs WAV audio from an input waveform. It does not
provide a compressed file format or entropy coder. The checkpoint, original
training corpus, and MUSHRA samples are not committed to this repository.

Code adapted from DAC remains under the MIT license in [LICENSE](LICENSE).
