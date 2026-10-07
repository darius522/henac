"""Exercise the installed inference and evaluation commands end to end."""

import csv
import json
import math
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf
import torch

from henac.model import DAC


@pytest.mark.parametrize("sample_rate,channels", [(32000, 1), (44100, 2)])
def test_reconstruct_and_evaluate_from_checkpoint(tmp_path, sample_rate, channels):
    model_config = {
        "stage": "high",
        "encoder_dim": 8,
        "latent_dim": 128,
        "encoder_rates": [2, 2, 5, 20],
        "decoder_rates": [20, 5, 2, 2],
        "n_codebooks": 2,
        "codebook_size": 8,
        "codebook_dim": 4,
        "sample_rate": 32000,
        "skip_args": {
            0: {"codebook_dim": 16, "n_codebooks": 1, "encoder_rates": [5], "decoder_rates": [5]},
            1: {
                "codebook_dim": 16,
                "n_codebooks": 1,
                "encoder_rates": [16],
                "decoder_rates": [16],
            },
        },
    }
    torch.manual_seed(7)
    model = DAC(**model_config)
    checkpoint = tmp_path / "model.pth"
    torch.save({"model_config": model_config, "model_state": model.state_dict()}, checkpoint)

    reference = tmp_path / "reference"
    reference.mkdir()
    samples = np.sin(2 * np.pi * 440 * np.arange(round(sample_rate * 0.025)) / sample_rate)
    if channels == 2:
        samples = np.column_stack((samples, 0.5 * samples))
    sf.write(reference / "tone.wav", samples.astype("float32"), sample_rate)
    reconstruction = tmp_path / "reconstruction"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "henac.infer",
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(reference),
            "--output",
            str(reconstruction),
            "--codebooks",
            "1",
            "1",
            "1",
            "--device",
            "cpu",
            "--chunk-seconds",
            "0.0125",
            "--context-seconds",
            "0",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    audio, rate = sf.read(reconstruction / "tone.wav")
    assert rate == 32000
    assert audio.shape == ((800,) if channels == 1 else (800, 2))
    with (reconstruction / "entropy.csv").open(newline="") as file:
        row = next(csv.DictReader(file))
    assert row["file"] == "tone.wav"
    assert float(row["total_kbps"]) >= 0

    subprocess.run(
        [
            sys.executable,
            "-m",
            "henac.evaluate",
            "--reference",
            str(reference),
            "--reconstruction",
            str(reconstruction),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((reconstruction / "evaluation.json").read_text())
    assert report["files"] == 1
    assert math.isfinite(report["full_l1"])
