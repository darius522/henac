import os
import sys
sys.path.append(os.getcwd())

import dac
from audiotools import AudioSignal

import pandas as pd
import torch
from tqdm import tqdm
import soundfile as sf
import numpy as np

from utils.audio_utils import resample_bands
from julius import resample_frac

from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

import argparse

DURATION = 1.

def indices_to_entropy(indices, time_axis=1, eps=1e-20, size=1024) -> torch.Tensor:
    n_step = indices.shape[time_axis]
    oh_indices = torch.nn.functional.one_hot(indices, num_classes=size)
    p = (torch.sum(oh_indices, dim=time_axis) + eps) / n_step
    return -torch.sum(torch.mul(p, torch.log(p)), axis=-1)  # * n_step


def main(args):
    # Download a model
    # model_path = dac.utils.download(model_type="24khz")
    model = dac.DAC.load(args.model_path)
    model.to("cuda:7")

    dataset = pd.read_csv(args.dataset)
    entropies, snrs = [], []

    for i, row in tqdm(dataset.iterrows(), total=len(dataset)):
        fname = os.path.basename(row.path).split(".")[0]
        # Load audio signal file
        signal = AudioSignal(row.path, duration=DURATION)

        # Encode audio signal as one long file
        # (may run out of GPU memory on long files)
        signal.to(model.device)

        x = model.preprocess(signal.audio_data, signal.sample_rate)
        z, codes, _, _, _, skips = model.encode(x)
        entropies.append(
            indices_to_entropy(codes.permute(0, 2, 1), time_axis=1, size=1024)
            .mean()
            .cpu()
            .item()
        )

        # Decode audio signal
        y: AudioSignal
        skips, _, _ = model.autoencode_skips(skips)
        y = model.multidecode(z, skips)
        y = [
            resample_frac(sig, int(sig.shape[-1] // DURATION), signal.sample_rate)[..., : int(signal.sample_rate * DURATION)]
            for sig in y
        ]
        y = torch.stack(y).sum(0)

        y, signal = y.to("cpu").detach(), signal.audio_data.to("cpu").detach()
        snrs.append(ScaleInvariantSignalNoiseRatio().to("cpu")(y, signal))

        sf.write(
            os.path.join(args.output_path, f"{fname}_full_i.wav"),
            signal.reshape(-1).numpy(),
            samplerate=24_000,
        )
        sf.write(
            os.path.join(args.output_path, f"{fname}_full_o.wav"),
            y.reshape(-1).numpy(),
            samplerate=24_000,
        )

    print(f"Overall Entropy: {np.round(np.mean(entropies), 1)}")
    print(f"Overall SNR: {np.round(np.mean(snrs), 1)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run compression on a dataset. Save the audio files"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/home/daripete/jstsp-dac/datasets/fma_test_subset.csv",
        required=False,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/daripete/jstsp-dac/runs/quant_1024_dropout=05_multidisc_32cb_hbonly/latest/dac/weights.pth",
        required=False,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/home/daripete/jstsp-dac/runs/quant_1024_dropout=05_multidisc_32cb_hbonly/audios",
        required=False,
    )
    args = parser.parse_args()

    main(args)
