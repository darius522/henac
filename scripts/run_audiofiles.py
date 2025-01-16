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

from matplotlib import pyplot as plt


def compute_entropy(tensor):
    B, I = tensor.shape
    entropies = torch.zeros(B, device=tensor.device)

    for b in range(B):
        unique_vals, counts = torch.unique(tensor[b], return_counts=True)  # Get unique values and their counts
        probs = counts.float() / I  # Compute probabilities
        entropy = -torch.sum(probs * torch.log2(probs + 1e-9))  # Compute entropy (adding small value for stability)
        entropies[b] = entropy

    return entropies  # Shape: [B]


def main(args):
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    model = dac.DAC.load(args.model_path)
    model.eval()
    model.to("cuda")

    dataset = pd.read_csv(args.dataset)
    entropies, snrs = [], dict(band=[], full=[])
    
    resampler = julius.SplitBands(24_000, cutoffs=[3000, 6000]).to('cuda')
    duration = 5.0
    for i, row in tqdm(dataset.iterrows(), total=len(dataset)):
        fname = os.path.basename(row.path).split('.')[0]
        # Load audio signal file
        signal = AudioSignal(row.path, duration=duration)

        # Encode audio signal as one long file
        # (may run out of GPU memory on long files)
        signal.to(model.device)

        out = model.forward(signal.audio_data, n_quantizers=None)
        y, codes = out['audio'], out['codes']
        #bands = resampler(signal.audio_data)
        entropies.append(compute_entropy(codes.squeeze(0)).detach().cpu().numpy())
        
        y_band, signal_band = resampler(y)[1:].sum(0), -resampler(signal.audio_data)[1:].sum(0)
        y_band = normalize_to_match_peak_batched(y_band, signal_band)
        diff = (y_band + signal_band).to('cpu').detach()

        y, signal = y.to('cpu').detach(), signal.audio_data.to('cpu').detach()
        y_band, signal_band = y_band.to('cpu').detach(), signal_band.to('cpu').detach()
        snrs['full'].append(ScaleInvariantSignalNoiseRatio().to("cpu")(y, signal))
        snrs['band'].append(ScaleInvariantSignalNoiseRatio().to("cpu")(y_band, signal_band))
        
        sf.write(
            os.path.join(args.output_path, f"{fname}_mid_input.wav"),
            signal_band.reshape(-1).numpy(),
            samplerate=24_000,
        )
        sf.write(
            os.path.join(args.output_path, f"{fname}_mid_output.wav"),
            y_band.reshape(-1).numpy(),
            samplerate=24_000,
        )

        sf.write(
            os.path.join(args.output_path, f"{fname}_mid_diff.wav"),
            diff.reshape(-1).numpy(),
            samplerate=24_000,
        )
        
        del out
        
    
    br_per_cb = np.array(entropies).mean((0,1))
    print(f'Overall Entropy: {np.round(br_per_cb, 1)}')
    for k, v in snrs.items():
        print(f'Overall SNR for {k}: {np.round(np.mean(v), 1)}')

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
        default="/N/slate/daripete/jstsp-dac/runs/midband_decfrozen_nodiff/latest/dac/weights.pth",
        required=False,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/N/slate/daripete/jstsp-dac/runs/midband_decfrozen_nodiff/latest/audios",
        required=False,
    )
    args = parser.parse_args()

    main(args)
