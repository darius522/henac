# Training parity with the three original branches

The consolidated trainer was checked against the original `baseline_skipae`,
`entropy_ctrl`, and `entropy_ctrl_hb` trainers. The compared revisions are
`003125f2b7d1dc172ed71cffcf090a4a1e70884d`,
`e211a41e6bd881dec6ef548644dfcac9cc944db0`, and
`1ae98408549d4e1d895d85d21124524ef634f97d`, respectively.

## Method

`scripts/verify_training_parity.py` runs each implementation in a separate
process so each can import its own `dac` package. Each stage receives the same
initial weights and first 0.38 seconds of
`checkpoints/mushra/input/001662_chunk_0.wav`. The check compares the
frequency band loss target, training waveform, code indices, discriminator and
generator losses, gradients, and all shared model and discriminator weights
after one AdamW update. The synthetic models have the same stage structure
with smaller dimensions. The final high band check strictly loads the supplied
300k generator and discriminator weights before the step.

In the synthetic core case, the original model also contains 64 unused
mid-band skip AE tensors. The consolidated core model omits them, so they are
excluded from the core weight comparison.

The comparison uses the legacy training loop itself, the consolidated
`train_step`, identical loss settings, and a single GPU with AMP disabled. It
holds the input waveform fixed and bypasses both data loaders to isolate the
model and optimizer calculations. Both trainers use the default cuDNN
benchmark setting. The original FMA audio files referenced by the branch
manifests are not available in this workspace.

## Measured result

| Stage | Waveform, target, codes, and losses | Active weight relative L2 | Gradient relative L2 | Largest weight difference |
| --- | --- | ---: | ---: | ---: |
| Core synthetic | Exact | `0` | `0` | `0` |
| Mid synthetic | Exact | `2.02e-6` | `2.61e-7` | `1.03e-4` |
| High synthetic | Exact | `2.99e-8` | `1.03e-7` | `9.87e-7` |
| Final 300k high | Exact | `5.55e-8` | `2.10e-5` | `1.83e-4` |

For the full 300k high band checkpoint, the waveform, target, all code paths,
and every compared loss term were identical. Across 675 generator tensors,
the updated weights had relative L2 difference `4.61e-8`; across 130 gradient
tensors, the relative L2 difference was `2.10e-5`. Across 324 discriminator
tensors, the updated weight difference was `3.94e-9` relative L2.

CUDA reflection padding backward is nondeterministic in the checked PyTorch
runtime. Rerunning the **legacy code against itself** gave relative L2
differences of `2.79e-8` for updated generator weights and `1.88e-5` for
gradients. Rerunning the consolidated code against itself gave `5.03e-8` and
`2.39e-5`, respectively. The cross implementation differences are on this
repeat-run scale; the generator and gradient differences are below the
consolidated repeat differences. This is numerical training step parity, not
bit exact backward parity. The synthetic mid update's largest individual
weight difference was `1.03e-4`, while its active-weight relative L2 was
`2.02e-6`; no mid-stage repeat baseline was measured.

The detached functional validation also completed core → mid → high one step
training, checkpoint resume, and two GPU distributed training, each with exit
status zero. Its log is `/tmp/henac-overnight-final-20261007/run.log` in the
checked workspace. The detached training parity run also exited zero; its
report files are in `/tmp/henac-training-parity-verified-20261007/results`.

After moving the installable package to `henac`, the one step comparison was
rerun against the same three pinned branches and the final 300k checkpoint.
All stage waveforms, band targets, code indices, and losses were exact. The
final high-band active-weight relative L2 difference was `4.55e-8`; the
gradient difference was `2.13e-5`. Two repeats of each implementation gave
active-weight differences of `5.44e-8` (legacy) and `5.11e-8` (consolidated),
and gradient differences of `2.33e-5` and `2.50e-5`. The rerun exited zero;
its reports are in `/tmp/henac-release-validation-20261007/training` in the
checked workspace.

A separately built, noneditable wheel also completed a one-step core → mid →
high training handoff with validation at each stage. All three checkpoints
were written and the run exited zero. Its logs are in
`/tmp/henac-release-validation-20261007/wheel-train`.

A second detached functional run used the full-size stage recipes and the
local MUSHRA WAVs to train core, mid, and high for three steps each. With a
two-example training dataset, the third step crossed the loader boundary.
Each stage validated at steps 2 and 3, and its saved checkpoint recorded step
3 with a finite validation score. The run exited zero; its logs and
checkpoints are in `/tmp/henac-training-sustained-20261007`.

## Reproduce

After installing the package and checking out the three pinned revisions,
run from the repository root on a CUDA GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/verify_training_parity.py run \
  --core-root ../baseline_skipae \
  --mid-root ../entropy_ctrl \
  --high-root ../henac-hb \
  --input checkpoints/mushra/input/001662_chunk_0.wav \
  --checkpoint-root checkpoints --repeat-final \
  --output-dir /tmp/henac-training-parity
```

The high band reference checkout can be created as shown in
[the inference parity record](reference-parity.md). The script checks all three
legacy revisions and writes per-stage JSON reports. It uses temporary files
outside Git for the large weight and gradient captures.

These one step checks and the synthetic stage handoffs do not establish parity
for a 400,000 step training trajectory. That would require the original
training audio, core and mid checkpoints, and the original distributed runtime.
