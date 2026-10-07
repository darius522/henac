"""Check the consolidated model against the supplied high-band reference."""

import argparse
import json
import sys
from pathlib import Path

import julius
import numpy as np
import soundfile as sf
import torch
import yaml
from audiotools import AudioSignal

from verify_mushra_reference import (
    CONFIG_SHA256,
    GOLD_MANIFEST_SHA256,
    INPUT_MANIFEST_SHA256,
    WEIGHTS_SHA256,
    compare,
    file_sha256,
    manifest_sha256,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--legacy-output-dir", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("This parity check requires a CUDA GPU")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    inputs = sorted((args.checkpoint_root / "mushra/input").glob("*.wav"))
    gold_dir = args.checkpoint_root / "mushra/output_[16, 4, 2]_23kbps_wild"
    golds = [gold_dir / path.name for path in inputs]
    expected = {
        "weights": (
            file_sha256(args.checkpoint_root / "300k/dac/weights.pth"),
            WEIGHTS_SHA256,
        ),
        "configuration": (file_sha256(args.checkpoint_root / "conf.yaml"), CONFIG_SHA256),
        "inputs": (manifest_sha256(inputs), INPUT_MANIFEST_SHA256),
        "gold WAVs": (manifest_sha256(golds), GOLD_MANIFEST_SHA256),
    }
    if len(inputs) != 39 or any(not path.is_file() for path in golds):
        parser.error("Expected all 39 paired reference WAV files")
    for name, (actual, wanted) in expected.items():
        if actual != wanted:
            raise ValueError(f"{name} hash mismatch: {actual} != {wanted}")

    from henac.model import DAC

    config = yaml.safe_load((args.checkpoint_root / "conf.yaml").read_text())
    model_args = {key[4:]: value for key, value in config.items() if key.startswith("DAC.")}
    model_args["stage"] = "high"
    model = DAC(**model_args)
    checkpoint = torch.load(
        args.checkpoint_root / "300k/dac/weights.pth",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.eval().to("cuda")
    splitter = julius.SplitBands(model.sample_rate, cutoffs=[3000, 6200]).to("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gold_results = []
    legacy_results = []
    for input_path, gold_path in zip(inputs, golds):
        signal = AudioSignal(str(input_path))
        if signal.sample_rate != model.sample_rate:
            raise ValueError(f"Unexpected sample rate: {input_path}")
        if signal.audio_data.shape[1] > 1:
            signal.audio_data = signal.audio_data.mean(dim=1, keepdim=True)
        signal.to("cuda")
        with torch.no_grad():
            paths = model.infer_bands(signal.audio_data, n_quantizers=[16, 4, 2])["audio"]
            waveform = np.zeros(signal.audio_data.shape)
            for band, name in enumerate(("core", "mb", "hb")):
                waveform += splitter(paths[name])[band].reshape(1, 1, -1).cpu().numpy()
        mono = waveform.reshape(-1)
        output_path = args.output_dir / input_path.name
        sf.write(
            output_path,
            np.column_stack((mono, mono)),
            model.sample_rate,
            subtype="PCM_16",
        )
        gold_results.append(compare(gold_path, output_path))
        if args.legacy_output_dir is not None:
            legacy_results.append(compare(args.legacy_output_dir / input_path.name, output_path))
        print(
            json.dumps(
                {
                    "file": input_path.name,
                    "gold": gold_results[-1],
                    "legacy": legacy_results[-1] if legacy_results else None,
                }
            ),
            flush=True,
        )

    summary = {
        "files": len(gold_results),
        "mean_exact_fraction": float(np.mean([r["exact_fraction"] for r in gold_results])),
        "min_exact_fraction": min(r["exact_fraction"] for r in gold_results),
        "min_snr_db": min(r["snr_db"] for r in gold_results),
        "legacy_exact": all(r["max_lsb"] == 0 for r in legacy_results) if legacy_results else None,
    }
    print(json.dumps({"summary": summary}), flush=True)
    if summary["min_exact_fraction"] < 0.985 or summary["min_snr_db"] < 45:
        raise SystemExit("Gold parity threshold failed")
    if legacy_results and not summary["legacy_exact"]:
        raise SystemExit("Consolidated output differs from legacy regenerated output")


if __name__ == "__main__":
    main()
