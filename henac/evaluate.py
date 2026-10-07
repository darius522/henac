"""Evaluate reconstructed audio against matching reference files."""

import argparse
import csv
import json
from pathlib import Path

import julius
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as audio_functional


def read_pair(reference_path: Path, reconstruction_path: Path):
    reference, reference_rate = sf.read(reference_path, dtype="float32", always_2d=True)
    reconstruction, sample_rate = sf.read(reconstruction_path, dtype="float32", always_2d=True)
    reference = torch.from_numpy(reference.T.copy())
    reconstruction = torch.from_numpy(reconstruction.T.copy())
    if reference_rate != sample_rate:
        reference = audio_functional.resample(reference, reference_rate, sample_rate)
    if reference.shape[0] != reconstruction.shape[0]:
        if reconstruction.shape[0] == 1:
            reference = reference.mean(0, keepdim=True)
        else:
            raise ValueError(f"Channel count differs for {reference_path.name}")
    difference = abs(reference.shape[-1] - reconstruction.shape[-1])
    if difference > 1:
        raise ValueError(f"Audio lengths differ by {difference} samples: {reference_path.name}")
    length = min(reference.shape[-1], reconstruction.shape[-1])
    return reference[..., :length], reconstruction[..., :length], sample_rate


def waveform_metrics(reference: torch.Tensor, reconstruction: torch.Tensor):
    error = reference - reconstruction
    energy = reference.square().sum(-1)
    noise = error.square().sum(-1)
    snr = 10 * torch.log10((energy + 1e-12) / (noise + 1e-12))

    centered = reference - reference.mean(-1, keepdim=True)
    estimate = reconstruction - reconstruction.mean(-1, keepdim=True)
    projection = (
        (estimate * centered).sum(-1, keepdim=True)
        * centered
        / (centered.square().sum(-1, keepdim=True) + 1e-12)
    )
    distortion = estimate - projection
    sisdr = 10 * torch.log10(
        (projection.square().sum(-1) + 1e-12) / (distortion.square().sum(-1) + 1e-12)
    )
    return {
        "l1": float(error.abs().mean()),
        "snr_db": float(snr.mean()),
        "sisdr_db": float(sisdr.mean()),
    }


def evaluate_pair(reference_path: Path, reconstruction_path: Path):
    reference, reconstruction, sample_rate = read_pair(reference_path, reconstruction_path)
    result = {"duration_seconds": reference.shape[-1] / sample_rate}
    bands = {"full": (reference, reconstruction)}
    if sample_rate > 12400:
        splitter = julius.SplitBands(sample_rate, cutoffs=[3000, 6200])
        xbands = splitter(reference[None, ...])
        ybands = splitter(reconstruction[None, ...])
        for index, name in enumerate(("core", "mid", "high")):
            bands[name] = (xbands[index, 0], ybands[index, 0])
    for name, (x, y) in bands.items():
        result.update(
            {f"{name}_{metric}": value for metric, value in waveform_metrics(x, y).items()}
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument(
        "--report", type=Path, help="CSV path; defaults to reconstruction/evaluation.csv"
    )
    args = parser.parse_args()
    if not args.reference.is_dir() or not args.reconstruction.is_dir():
        parser.error("reference and reconstruction must be directories")
    files = sorted(p for p in args.reference.rglob("*") if p.suffix.lower() in (".wav", ".flac"))
    if not files:
        parser.error("No reference WAV or FLAC files found")
    pairs = []
    for reference in files:
        relative = reference.relative_to(args.reference).with_suffix(".wav")
        reconstruction = args.reconstruction / relative
        if not reconstruction.is_file():
            raise FileNotFoundError(f"Missing reconstruction: {reconstruction}")
        pairs.append((reference, reconstruction, relative))

    rows = []
    for reference, reconstruction, relative in pairs:
        row = {"file": str(relative), **evaluate_pair(reference, reconstruction)}
        rows.append(row)
        print(f"evaluated {relative}", flush=True)
    report = args.report or args.reconstruction / "evaluation.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "files": len(rows),
        **{
            name: float(np.mean([row[name] for row in rows]))
            for name in rows[0]
            if name not in ("file", "duration_seconds")
        },
    }
    summary_path = report.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
