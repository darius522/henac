"""Reconstruct WAV or FLAC files with a HENAC checkpoint."""

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as audio_functional
from audiotools import AudioSignal

from henac.inference import (
    aligned_length,
    codebook_limits,
    empirical_bitrate,
    load_model,
    reconstruct,
)


EXTENSIONS = {".wav", ".flac"}


def input_files(path: Path) -> list[Path]:
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in EXTENSIONS)
    else:
        raise FileNotFoundError(path)
    if not files:
        raise ValueError(f"No WAV or FLAC files found in {path}")
    return files


def reconstruct_file(model, audio: torch.Tensor, counts, chunk_seconds, context_seconds):
    chunk = round(chunk_seconds * model.sample_rate)
    context = round(context_seconds * model.sample_rate)
    if chunk <= 0 or context < 0:
        raise ValueError("chunk_seconds must be positive and context_seconds nonnegative")
    if audio.shape[-1] <= chunk:
        return reconstruct(model, audio, counts)

    waveforms = []
    path_codes = {}
    total = audio.shape[-1]
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        left = max(0, start - context)
        right = min(total, end + context)
        part, codes = reconstruct(model, audio[..., left:right], counts)
        waveforms.append(part[start - left : end - left])

        # Keep code frames belonging to the retained audio, excluding context.
        padded = aligned_length(model, right - left, counts)
        for name, values in codes.items():
            frames = values.shape[-1]
            first = round((start - left) * frames / padded)
            last = round((end - left) * frames / padded)
            path_codes.setdefault(name, []).append(values[..., first:last].cpu())
    return np.concatenate(waveforms), {
        name: torch.cat(parts, dim=-1) for name, parts in path_codes.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, help="Legacy checkpoint configuration, if not next to weights"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codebooks", type=int, nargs=3, metavar=("CORE", "MID", "HIGH"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-seconds", type=float, default=10.0)
    parser.add_argument("--context-seconds", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.device.startswith("cuda"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    model = load_model(args.checkpoint, device=args.device, config=args.config)
    counts = tuple(args.codebooks) if args.codebooks else codebook_limits(model)
    files = input_files(args.input)
    destinations = []
    for source in files:
        relative = source.relative_to(args.input) if args.input.is_dir() else Path(source.name)
        destinations.append(args.output / relative.with_suffix(".wav"))
    if len(destinations) != len(set(destinations)):
        raise ValueError("Input files would produce duplicate output WAV names")
    existing = [path for path in destinations if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output already exists: {existing[0]}; use --overwrite")

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for source, destination in zip(files, destinations):
        signal = AudioSignal(str(source))
        channels = signal.audio_data.shape[1]
        waveform_data = signal.audio_data
        if signal.sample_rate != model.sample_rate:
            waveform_data = audio_functional.resample(
                waveform_data, signal.sample_rate, model.sample_rate
            )
        if channels > 1:
            waveform_data = waveform_data.mean(dim=1, keepdim=True)
        audio = waveform_data.to(args.device)
        waveform, codes = reconstruct_file(
            model, audio, counts, args.chunk_seconds, args.context_seconds
        )
        duration = len(waveform) / model.sample_rate
        bitrate = empirical_bitrate(codes, duration, model.codebook_size)
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            destination,
            np.repeat(waveform[:, None], channels, axis=1),
            model.sample_rate,
            subtype="PCM_16",
        )
        rows.append(
            {
                "file": str(source.relative_to(args.input)) if args.input.is_dir() else source.name,
                "duration_seconds": duration,
                **{
                    f"{name}_kbps": bitrate.get(name, 0.0) / 1000
                    for name in ("core", "mb", "hb", "total")
                },
            }
        )
        print(
            f"{source} -> {destination} ({bitrate['total'] / 1000:.2f} estimated kbps)", flush=True
        )
    with (args.output / "entropy.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
