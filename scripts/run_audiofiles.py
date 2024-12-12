import sys

sys.path.append('/home/daripete/jstsp-dac')

import dac
from audiotools import AudioSignal

import pandas as pd
import torch
from tqdm import tqdm
import soundfile as sf 
import numpy as np

from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

import argparse, os


def indices_to_entropy(indices, time_axis=1, eps=1e-20, size=1024) -> torch.Tensor:
    n_step = indices.shape[time_axis]
    oh_indices  = torch.nn.functional.one_hot(indices, num_classes=size)
    p = (torch.sum(oh_indices, dim=time_axis) + eps) / n_step
    return -torch.sum(torch.mul(p, torch.log(p)), axis=-1)  # * n_step


def main(args):
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    # Download a model
    # model_path = dac.utils.download(model_type="24khz")
    model = dac.DAC.load(args.model_path)
    model.to("cuda")

    dataset = pd.read_csv(args.dataset)
    entropies, snrs = [], []

    for i, row in tqdm(dataset.iterrows(), total=len(dataset)):
        fname = os.path.basename(row.path).split('.')[0]
        # Load audio signal file
        signal = AudioSignal(row.path, duration=10.0)

        # Encode audio signal as one long file
        # (may run out of GPU memory on long files)
        signal.to(model.device)

        x = model.preprocess(signal.audio_data, signal.sample_rate)
        z, codes, latents, _, _ = model.encode(x)
        entropies.append(indices_to_entropy(
            codes.permute(0, 2, 1), time_axis=1, size=1024
        ).mean().cpu().item())

        # Decode audio signal
        y: AudioSignal
        y = model.decode(z)

        y, signal = y.to('cpu').detach(), signal.audio_data.to('cpu').detach()
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

    print(f'Overall Entropy: {np.round(np.mean(entropies), 1)}')
    print(f'Overall SNR: {np.round(np.mean(snrs), 1)}')

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
        default="/home/daripete/jstsp-dac/runs/sanity_skip_1/50k/dac/weights.pth",
        required=False,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/home/daripete/jstsp-dac/runs/sanity_skip_1/",
        required=False,
    )
    args = parser.parse_args()

    main(args)
