"""Compare the legacy HB model against the supplied MUSHRA reference WAVs.

The model code is imported from an isolated checkout of entropy_ctrl_hb so this
check remains usable while the main branch is being consolidated.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import julius
import numpy as np
import soundfile as sf
import torch
import yaml
from audiotools import AudioSignal

REFERENCE_REVISION = "1ae98408549d4e1d895d85d21124524ef634f97d"
WEIGHTS_SHA256 = "4e511c753eeed87d251bc0e6d33f5e50dc8be0cd135570c6b3acd92db2cdc526"
CONFIG_SHA256 = "07e8e40956949894ef2c0bf391b08ed0637d7635ada64c32f528fe878722964c"
INPUT_MANIFEST_SHA256 = "7e26cf03c60a4a74e8b0b4327f51cff260f6750cbc98d7d496a4578c7c071962"
GOLD_MANIFEST_SHA256 = "cca25ace0ed80a47f1b113baab485365998a52e88ad2ae3bbf093c87fb13683d"


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_sha256(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update((file_sha256(path) + "\n").encode())
    return digest.hexdigest()


def verify_reference_assets(args, input_paths, gold_paths):
    revision = subprocess.check_output(
        ["git", "-C", str(args.reference_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    expected = {
        "branch revision": (revision, REFERENCE_REVISION),
        "weights": (file_sha256(args.checkpoint_root / "300k/dac/weights.pth"), WEIGHTS_SHA256),
        "configuration": (file_sha256(args.checkpoint_root / "conf.yaml"), CONFIG_SHA256),
        "inputs": (manifest_sha256(input_paths), INPUT_MANIFEST_SHA256),
        "gold WAVs": (manifest_sha256(gold_paths), GOLD_MANIFEST_SHA256),
    }
    for label, (actual, wanted) in expected.items():
        if actual != wanted:
            raise ValueError(f"{label} SHA/revision mismatch: {actual} != {wanted}")


def load_model(reference_root: Path, checkpoint_root: Path):
    sys.path.insert(0, str(reference_root.resolve()))
    import dac

    with (checkpoint_root / "conf.yaml").open() as file:
        config = yaml.safe_load(file)
    model_args = {
        key.removeprefix("DAC."): value for key, value in config.items() if key.startswith("DAC.")
    }
    model = dac.DAC.load(
        str(checkpoint_root / "300k/dac/weights.pth"),
        strict=True,
        **model_args,
    )
    return model.eval().to("cuda")


def reproduce(model, splitter, input_path: Path, output_path: Path):
    signal = AudioSignal(str(input_path))
    if signal.sample_rate != model.sample_rate:
        raise ValueError(f"Unexpected sample rate in {input_path}")
    if signal.audio_data.shape[1] > 1:
        signal.audio_data = signal.audio_data.mean(dim=1, keepdim=True)
    signal.to("cuda")

    with torch.no_grad():
        paths = model.infer_bands(signal.audio_data, n_quantizers=[16, 4, 2])["audio"]
        waveform = np.zeros(signal.audio_data.shape)
        for band, name in enumerate(("core", "mb", "hb")):
            filtered = splitter(paths[name])[band]
            waveform += filtered.reshape(1, 1, -1).cpu().numpy()

    mono = waveform.reshape(-1)
    sf.write(
        output_path,
        np.column_stack((mono, mono)),
        model.sample_rate,
        subtype="PCM_16",
    )


def compare(gold_path: Path, output_path: Path):
    gold, gold_rate = sf.read(gold_path, dtype="int16", always_2d=True)
    output, output_rate = sf.read(output_path, dtype="int16", always_2d=True)
    if gold_rate != output_rate or gold.shape != output.shape:
        raise ValueError(f"WAV shape or sample rate differs: {gold_path.name}")

    error = output.astype(np.int32) - gold.astype(np.int32)
    rmse = float(np.sqrt(np.mean(error.astype(np.float64) ** 2)))
    gold_rms = float(np.sqrt(np.mean(gold.astype(np.float64) ** 2)))
    return {
        "file": gold_path.name,
        "exact_fraction": float(np.mean(error == 0)),
        "over_one_lsb_fraction": float(np.mean(np.abs(error) > 1)),
        "max_lsb": int(np.max(np.abs(error))),
        "snr_db": float(20 * np.log10(gold_rms / rmse)) if rmse else float("inf"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-snr-db", type=float, default=45.0)
    parser.add_argument("--min-exact-fraction", type=float, default=0.985)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("This reference check requires a CUDA GPU")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    inputs = args.checkpoint_root / "mushra/input"
    golds = args.checkpoint_root / "mushra/output_[16, 4, 2]_23kbps_wild"
    input_paths = sorted(inputs.glob("*.wav"))
    gold_paths = [golds / path.name for path in input_paths]
    if len(input_paths) != 39 or any(not path.is_file() for path in gold_paths):
        parser.error("Expected all 39 paired reference WAV files")
    verify_reference_assets(args, input_paths, gold_paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.reference_root, args.checkpoint_root)
    splitter = julius.SplitBands(model.sample_rate, cutoffs=[3000, 6200]).to("cuda")
    results = []
    for input_path, gold_path in zip(input_paths, gold_paths):
        output_path = args.output_dir / input_path.name
        reproduce(model, splitter, input_path, output_path)
        result = compare(gold_path, output_path)
        print(json.dumps(result), flush=True)
        results.append(result)

    summary = {
        "files": len(results),
        "mean_exact_fraction": float(np.mean([r["exact_fraction"] for r in results])),
        "min_exact_fraction": min(r["exact_fraction"] for r in results),
        "min_snr_db": min(r["snr_db"] for r in results),
        "files_over_one_lsb": sum(r["max_lsb"] > 1 for r in results),
    }
    print(json.dumps({"summary": summary}), flush=True)
    if (
        summary["min_snr_db"] < args.min_snr_db
        or summary["min_exact_fraction"] < args.min_exact_fraction
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
