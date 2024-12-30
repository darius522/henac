import sys

sys.path.append('/N/slate/daripete/jstsp-dac')

import dac
from audiotools import AudioSignal

import pandas as pd
import torch
from tqdm import tqdm
import soundfile as sf 
import numpy as np

from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

import argparse, os

from utils.audio_utils import normalize_to_match_peak_batched

import julius


def indices_to_entropy(indices, time_axis=1, eps=1e-20, size=1024) -> torch.Tensor:
    n_step = indices.shape[time_axis]
    oh_indices  = torch.nn.functional.one_hot(indices, num_classes=size)
    p = (torch.sum(oh_indices, dim=time_axis) + eps) / n_step
    return -torch.sum(torch.mul(p, torch.log(p)), axis=-1) * n_step


def main(args):
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    # Download a model
    # model_path = dac.utils.download(model_type="24khz")
    model = dac.DAC.load(args.model_path)
    model.eval()
    model.to("cuda")

    dataset = pd.read_csv(args.dataset)
    entropies, snrs = [], []
    
    resampler = julius.SplitBands(24_000, cutoffs=[3000, 6000]).to('cuda')

    for i, row in tqdm(dataset.iterrows(), total=len(dataset)):
        fname = os.path.basename(row.path).split('.')[0]
        # Load audio signal file
        signal = AudioSignal(row.path, duration=5.0)

        # Encode audio signal as one long file
        # (may run out of GPU memory on long files)
        signal.to(model.device)

        out = model.forward(signal.audio_data, n_quantizers=None)
        y, codes = out['audio'], out['codes']
        bands = resampler(signal.audio_data)
        entropies.append(indices_to_entropy(
            codes.permute(0, 2, 1), time_axis=1, size=1024
        ).detach().cpu().numpy())

        y, signal = y.to('cpu').detach(), signal.audio_data.to('cpu').detach()
        # y = normalize_to_match_peak_batched(y, signal)
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
        
        del out
        
    
    br_per_cb = np.array(entropies).mean((0,1))
    print(f'Overall Entropy: {np.round(br_per_cb, 1)}')
    print(f'Overall SNR: {np.round(np.mean(snrs), 1)}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run compression on a dataset. Save the audio files"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/N/slate/daripete/jstsp-dac/datasets/fma_test_subset.csv",
        required=False,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/N/slate/daripete/jstsp-dac/runs/_dummy/latest/dac/weights.pth",
        required=False,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/N/slate/daripete/jstsp-dac/runs/_dummy/audios",
        required=False,
    )
    args = parser.parse_args()

    main(args)
