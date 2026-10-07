"""Compare one training step against the three pinned legacy branches.

The ``run`` command orchestrates isolated processes because every legacy
branch exposes a different implementation under the same ``dac`` package name.
This check needs the local MUSHRA input and the three original checkouts.
"""

import argparse
import contextlib
import json
import math
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch


REPO = Path(__file__).resolve().parents[1]
REVISIONS = {
    "core": "003125f2b7d1dc172ed71cffcf090a4a1e70884d",
    "mid": "e211a41e6bd881dec6ef548644dfcac9cc944db0",
    "high": "1ae98408549d4e1d895d85d21124524ef634f97d",
}
SKIP_ARGS = {
    0: {
        "codebook_dim": 16,
        "n_codebooks": 1,
        "encoder_rates": [5],
        "decoder_rates": [5],
        "quantizer_dropout": 0.0,
        "gumbel_softmax": False,
        "diff_entropy": False,
    },
    1: {
        "codebook_dim": 16,
        "n_codebooks": 1,
        "encoder_rates": [16],
        "decoder_rates": [16],
        "quantizer_dropout": 0.0,
        "gumbel_softmax": False,
        "diff_entropy": False,
    },
}
LAMBDA_WEIGHTS = {
    "mel/loss": 15.0,
    "adv/feat_loss": 2.0,
    "adv/gen_loss": 1.0,
    "vq/commitment_loss": 0.25,
    "vq/codebook_loss": 1.0,
}
LOSS_NAMES = (
    "loss",
    "mel/loss",
    "adv/disc_loss",
    "adv/feat_loss",
    "adv/gen_loss",
    "vq/commitment_loss",
    "vq/codebook_loss",
    "vq/entropy_loss",
)


class _NoScale:
    def unscale_(self, optimizer):
        return None


class _SingleGPU:
    device = torch.device("cuda")
    world_size = 1
    scaler = _NoScale()

    def autocast(self):
        return contextlib.nullcontext()

    def backward(self, loss):
        loss.backward()

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None


class _LegacyWrapper(torch.nn.Module):
    """Expose ``.module`` as the legacy trainer expects from DDP."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _check_revision(stage: str, root: Path):
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != REVISIONS[stage]:
        raise ValueError(f"{stage} legacy revision is {actual}; expected {REVISIONS[stage]}")


def _model_arguments(stage: str, legacy: bool, checkpoint_root: Path | None):
    if checkpoint_root is not None:
        import yaml

        config = yaml.safe_load((checkpoint_root / "conf.yaml").read_text())
        model = {key[4:]: value for key, value in config.items() if key.startswith("DAC.")}
        discriminator = {
            key.removeprefix("Discriminator."): value
            for key, value in config.items()
            if key.startswith("Discriminator.")
        }
        return model, discriminator

    model = {
        "encoder_dim": 8,
        "latent_dim": 128,
        "encoder_rates": [2, 2, 5, 20],
        "decoder_rates": [20, 5, 2, 2],
        "n_codebooks": 2,
        "codebook_size": 8,
        "codebook_dim": 4,
        "quantizer_dropout": 0.0,
        "sample_rate": 32000,
        "min_n_codebooks": 1,
    }
    if legacy:
        if stage == "high":
            model["skip_args"] = deepcopy(SKIP_ARGS)
        elif stage == "mid":
            model["skip_args"] = dict(SKIP_ARGS[0], decoder_dim=128)
        else:
            model["skip_args"] = deepcopy(SKIP_ARGS[0])
    else:
        model["skip_args"] = deepcopy(SKIP_ARGS)
    discriminator = {
        "sample_rate": 32000,
        "rates": [],
        "periods": [2, 3],
        "fft_sizes": [512],
        "bands": [[0.0, 0.1], [0.1, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1.0]],
    }
    return model, discriminator


def _freeze_legacy(model, stage: str):
    if stage == "core":
        return
    index = 0 if stage == "mid" else 1
    active = (f"skip_aes.{index}", f"multidecoders.{index}")
    blind = f"multidecoders.{index}.model.0.t_conv.blind."
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(any(prefix in name for prefix in active) and blind not in name)


def _batch(input_path: Path):
    import numpy as np
    import soundfile as sf
    from audiotools import AudioSignal

    samples, rate = sf.read(input_path, dtype="float32", always_2d=True)
    if rate != 32000 or len(samples) < 12160:
        raise ValueError("Training parity input must have at least 0.38 seconds at 32 kHz")
    mono = samples[:12160].mean(axis=1)
    audio = torch.from_numpy(np.asarray(mono)[None, None, :].copy())
    return {"signal": AudioSignal(audio, rate), "transform_args": {"keep": torch.tensor(0)}}


def capture(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Training parity capture requires a CUDA GPU")
    if args.checkpoint_root is not None and args.stage != "high":
        raise ValueError("The supplied final checkpoint is high band")
    if args.implementation == "legacy":
        if args.legacy_root is None:
            raise ValueError("A legacy capture requires --legacy-root")
        _check_revision(args.stage, args.legacy_root)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(20261007)

    source_root = args.legacy_root if args.implementation == "legacy" else REPO
    sys.path.insert(0, str(source_root.resolve()))
    import julius

    legacy = args.implementation == "legacy"
    if legacy:
        import dac
        from scripts import train as trainer

        codec_model = dac.model
        codec_losses = dac.nn.loss
    else:
        from henac import model as codec_model
        from henac.nn import loss as codec_losses
        from henac.training_stage import configure_trainable
        from henac import train as trainer

    model_args, disc_args = _model_arguments(args.stage, legacy, args.checkpoint_root)
    model = codec_model.DAC(**model_args, **({} if legacy else {"stage": args.stage}))
    discriminator = codec_model.Discriminator(**disc_args)
    if args.checkpoint_root is not None:
        for module, relative_path in (
            (model, "300k/dac/weights.pth"),
            (discriminator, "300k/discriminator/weights.pth"),
        ):
            saved = torch.load(
                args.checkpoint_root / relative_path, map_location="cpu", weights_only=True
            )
            module.load_state_dict(saved["state_dict"], strict=True)
    elif legacy:
        args.initial.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": _cpu_state(model), "discriminator": _cpu_state(discriminator)}, args.initial
        )
    else:
        initial = torch.load(args.initial, map_location="cpu", weights_only=True)
        selected = {
            key: value for key, value in initial["model"].items() if key in model.state_dict()
        }
        model.load_state_dict(selected, strict=True)
        discriminator.load_state_dict(initial["discriminator"], strict=True)

    if legacy:
        _freeze_legacy(model, args.stage)
    else:
        configure_trainable(model)
    model = model.to("cuda")
    discriminator = discriminator.to("cuda")
    generator = _LegacyWrapper(model) if legacy else model
    optimizer_g = torch.optim.AdamW(
        (parameter for parameter in generator.parameters() if parameter.requires_grad),
        lr=1e-4,
        betas=(0.8, 0.99),
        weight_decay=0.01,
    )
    optimizer_d = torch.optim.AdamW(
        discriminator.parameters(), lr=1e-4, betas=(0.8, 0.99), weight_decay=0.01
    )
    splitter = julius.SplitBands(32000, cutoffs=[3000, 6000]).to("cuda")
    batch = _batch(args.input)
    band = {"core": 0, "mid": 1, "high": 2}[args.stage]
    target = splitter(batch["signal"].audio_data.to("cuda"))[band:].sum(0)
    mel_min = [0 if band == 0 else 3000 if band == 1 else 6000] * 7
    losses = codec_losses
    state = SimpleNamespace(
        generator=generator,
        discriminator=discriminator,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        scheduler_g=torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=0.999996),
        scheduler_d=torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=0.999996),
        stft_loss=losses.MultiScaleSTFTLoss(window_lengths=[2048, 512]),
        mel_loss=losses.MelSpectrogramLoss(
            n_mels=[5, 10, 20, 40, 80, 160, 320],
            window_lengths=[32, 64, 128, 256, 512, 1024, 2048],
            mel_fmin=mel_min,
            mel_fmax=[None] * 7,
            pow=1.0,
            clamp_eps=1e-5,
            mag_weight=0.0,
        ),
        gan_loss=losses.GANLoss(discriminator),
        waveform_loss=losses.L1Loss(),
        train_data=SimpleNamespace(transform=lambda signal, **kwargs: signal),
        tracker=SimpleNamespace(step=0),
        resampler=splitter,
        splitter=splitter,
        stage=args.stage,
        scaler=1.0,
    )
    model.train()
    discriminator.train()
    torch.manual_seed(4444)
    with torch.no_grad():
        before = model(batch["signal"].audio_data.to("cuda"), sample_rate=32000, step=0)
    torch.manual_seed(4444)
    if legacy:
        if args.stage == "core":
            metrics = trainer.train_loop(
                state, batch, _SingleGPU(), LAMBDA_WEIGHTS, str(args.output.parent)
            )
        else:
            metrics = trainer.train_loop(state, batch, _SingleGPU(), LAMBDA_WEIGHTS)
    else:
        metrics = trainer.train_step(state, batch, _SingleGPU(), LAMBDA_WEIGHTS)

    codes = before["codes"]
    if not isinstance(codes, dict):
        codes = {"core" if args.stage == "core" else "mb": codes}
        if args.stage == "mid":
            codes["core"] = before["core_codes"]
    result = {
        "target": target.detach().cpu(),
        "waveform": before["audio"].detach().cpu(),
        "codes": {name: value.detach().cpu() for name, value in codes.items()},
        "metrics": {name: float(value) for name, value in metrics.items() if name in LOSS_NAMES},
        "model": _cpu_state(model),
        "discriminator": _cpu_state(discriminator),
        "gradients": {
            name: parameter.grad.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(f"{args.implementation} {args.stage}: loss={result['metrics']['loss']:.8f}")


def _tensor_stats(reference, candidate, allowed_reference_only=()):
    extra = candidate.keys() - reference.keys()
    missing = {
        name
        for name in reference.keys() - candidate.keys()
        if not name.startswith(allowed_reference_only)
    }
    if extra or missing:
        raise ValueError(
            f"Tensor keys differ: unexpected={sorted(extra)}, missing={sorted(missing)}"
        )
    squared_error = 0.0
    squared_reference = 0.0
    worst = (None, 0.0)
    for name, value in candidate.items():
        expected = reference[name]
        if expected.shape != value.shape:
            raise ValueError(f"Tensor shape differs: {name}")
        if expected.dtype != value.dtype:
            raise ValueError(f"Tensor dtype differs: {name}")
        if not value.is_floating_point():
            if not torch.equal(expected, value):
                raise ValueError(f"Integer tensor differs: {name}")
            continue
        delta = (expected - value).to(torch.float64)
        error = float(delta.abs().max())
        if not math.isfinite(error):
            raise ValueError(f"Non-finite tensor difference: {name}")
        if error > worst[1]:
            worst = (name, error)
        squared_error += float(delta.square().sum())
        squared_reference += float(expected.to(torch.float64).square().sum())
    relative_l2 = (squared_error / squared_reference) ** 0.5 if squared_reference else 0.0
    if not math.isfinite(relative_l2):
        raise ValueError("Non-finite relative L2 difference")
    return {
        "tensors": len(candidate),
        "max_absolute_error": {"name": worst[0], "value": worst[1]},
        "relative_l2": relative_l2,
    }


def _compare_results(reference, candidate):
    if reference["codes"].keys() != candidate["codes"].keys():
        raise ValueError("Code paths differ")
    if reference["metrics"].keys() != candidate["metrics"].keys():
        raise ValueError("Training loss terms differ")
    if reference["gradients"].keys() != candidate["gradients"].keys():
        raise ValueError("Trainable parameter gradients differ")
    for name, value in reference["metrics"].items():
        if not math.isfinite(value) or not math.isfinite(candidate["metrics"][name]):
            raise ValueError(f"Non-finite training loss: {name}")
    active_keys = candidate["gradients"].keys()
    active_reference = {name: reference["model"][name] for name in active_keys}
    active_candidate = {name: candidate["model"][name] for name in active_keys}
    # The original core model carries an unused mid-band skip AE. The core
    # implementation omits only those 64 dormant checkpoint tensors.
    core_unused = ("skip_aes.0.",) if set(candidate["codes"]) == {"core"} else ()
    result = {
        "waveform_max_error": float((reference["waveform"] - candidate["waveform"]).abs().max()),
        "target_max_error": float((reference["target"] - candidate["target"]).abs().max()),
        "codes_equal": all(
            torch.equal(values, candidate["codes"][name])
            for name, values in reference["codes"].items()
        ),
        "loss_max_error": max(
            abs(value - candidate["metrics"][name]) for name, value in reference["metrics"].items()
        ),
        "model": _tensor_stats(reference["model"], candidate["model"], core_unused),
        "active_model": _tensor_stats(active_reference, active_candidate),
        "discriminator": _tensor_stats(reference["discriminator"], candidate["discriminator"]),
        "gradients": _tensor_stats(reference["gradients"], candidate["gradients"]),
    }
    for name in ("waveform_max_error", "target_max_error", "loss_max_error"):
        if not math.isfinite(result[name]):
            raise ValueError(f"Non-finite parity result: {name}")
    return result


def compare(args):
    if (args.repeat_reference is None) != (args.repeat_candidate is None):
        raise ValueError("Provide both repeat captures or neither")
    paths = (args.reference, args.candidate, args.repeat_reference, args.repeat_candidate)
    loaded = [
        torch.load(path, map_location="cpu", weights_only=True) if path is not None else None
        for path in paths
    ]
    result = _compare_results(loaded[0], loaded[1])
    if loaded[2] is not None and loaded[3] is not None:
        result["legacy_repeat"] = _compare_results(loaded[0], loaded[2])
        result["consolidated_repeat"] = _compare_results(loaded[1], loaded[3])
    report = json.dumps(result, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report)
    print(report, end="")
    if result["waveform_max_error"] or result["target_max_error"] or not result["codes_equal"]:
        raise AssertionError("Forward waveform, target, or codes differ")
    if result["loss_max_error"] > 1e-4:
        raise AssertionError("Training losses differ")
    bounds = {"model": 1e-5, "active_model": 1e-5, "discriminator": 1e-6, "gradients": 1e-4}
    for name, bound in bounds.items():
        if result[name]["relative_l2"] > bound:
            raise AssertionError(f"{name} relative L2 exceeds {bound}")


def run(args):
    if args.repeat_final and args.checkpoint_root is None:
        raise ValueError("--repeat-final requires --checkpoint-root")
    args.output_dir = args.output_dir.resolve()
    args.input = args.input.resolve()
    if args.checkpoint_root is not None:
        args.checkpoint_root = args.checkpoint_root.resolve()
    roots = {
        "core": args.core_root.resolve(),
        "mid": args.mid_root.resolve(),
        "high": args.high_root.resolve(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    for stage, root in roots.items():
        prefix = args.output_dir / stage
        initial = prefix.with_name(f"{stage}-initial.pth")
        legacy = prefix.with_name(f"{stage}-legacy.pth")
        consolidated = prefix.with_name(f"{stage}-consolidated.pth")
        for implementation, output in (("legacy", legacy), ("consolidated", consolidated)):
            command = [
                sys.executable,
                str(script),
                "capture",
                "--stage",
                stage,
                "--implementation",
                implementation,
                "--input",
                str(args.input.resolve()),
                "--initial",
                str(initial),
                "--output",
                str(output),
            ]
            if implementation == "legacy":
                command += ["--legacy-root", str(root.resolve())]
            subprocess.run(command, check=True, cwd=REPO)
        subprocess.run(
            [
                sys.executable,
                str(script),
                "compare",
                "--reference",
                str(legacy),
                "--candidate",
                str(consolidated),
                "--report",
                str(prefix.with_suffix(".json")),
            ],
            check=True,
            cwd=REPO,
        )
    if args.checkpoint_root is not None:
        prefix = args.output_dir / "final-high"
        files = {}
        for implementation in ("legacy", "consolidated"):
            variants = ("", "-repeat") if args.repeat_final else ("",)
            for variant in variants:
                output = prefix.with_name(f"final-high-{implementation}{variant}.pth")
                files[f"{implementation}{variant}"] = output
                subprocess.run(
                    [
                        sys.executable,
                        str(script),
                        "capture",
                        "--stage",
                        "high",
                        "--implementation",
                        implementation,
                        "--legacy-root",
                        str(roots["high"]),
                        "--input",
                        str(args.input),
                        "--initial",
                        str(prefix),
                        "--output",
                        str(output),
                        "--checkpoint-root",
                        str(args.checkpoint_root),
                    ],
                    check=True,
                    cwd=REPO,
                )
        command = [
            sys.executable,
            str(script),
            "compare",
            "--reference",
            str(files["legacy"]),
            "--candidate",
            str(files["consolidated"]),
            "--report",
            str(prefix.with_suffix(".json")),
        ]
        if args.repeat_final:
            command += [
                "--repeat-reference",
                str(files["legacy-repeat"]),
                "--repeat-candidate",
                str(files["consolidated-repeat"]),
            ]
        subprocess.run(command, check=True, cwd=REPO)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--stage", choices=REVISIONS, required=True)
    capture_parser.add_argument(
        "--implementation", choices=("legacy", "consolidated"), required=True
    )
    capture_parser.add_argument("--legacy-root", type=Path)
    capture_parser.add_argument("--input", type=Path, required=True)
    capture_parser.add_argument("--initial", type=Path, required=True)
    capture_parser.add_argument("--output", type=Path, required=True)
    capture_parser.add_argument("--checkpoint-root", type=Path)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--reference", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--repeat-reference", type=Path)
    compare_parser.add_argument("--repeat-candidate", type=Path)
    compare_parser.add_argument("--report", type=Path)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--core-root", type=Path, required=True)
    run_parser.add_argument("--mid-root", type=Path, required=True)
    run_parser.add_argument("--high-root", type=Path, required=True)
    run_parser.add_argument("--input", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--checkpoint-root", type=Path)
    run_parser.add_argument("--repeat-final", action="store_true")
    args = parser.parse_args()
    {"capture": capture, "compare": compare, "run": run}[args.command](args)


if __name__ == "__main__":
    main()
