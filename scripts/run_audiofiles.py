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


def compute_entropy(code_tensor, N=1024, M=4, frame_rate=500):
    """
    Calculate the entropy-based bitrate of an RVQ-based neural audio codec.
    
    Args:
    - code_tensor (ndarray): The code tensor of shape [M, T] where M is the number of codebooks and T is the number of time frames.
    - N (int): The size of each codebook (number of entries in the codebook).
    - M (int): The number of codebooks.
    - frame_rate (int): The frame rate (frames per second).
    
    Returns:
    - bitrate (float): The entropy-based bitrate in bits per second.
    """
    import numpy as np
    from scipy.stats import entropy
    # Initialize the total entropy
    total_entropy, total_bitrate = [], []
    
    # Loop over each codebook (axis 0)
    for i in range(M):
        # Get the codebook indices for this codebook (shape [T,])
        codebook_indices = code_tensor[i]
        
        # Compute the frequency distribution of the indices in the codebook
        # Calculate probabilities (relative frequencies)
        counts = np.bincount(codebook_indices, minlength=N)
        probabilities = counts / len(codebook_indices)
        
        # Calculate the entropy for this codebook
        codebook_entropy = entropy(probabilities, base=2)  # Entropy in bits
        
        # Add the entropy for this codebook to the total entropy
        total_entropy.append(codebook_entropy)
        total_bitrate.append(codebook_entropy*frame_rate)
    
    return total_bitrate


def main(args):
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    model = dac.DAC.load(args.model_path, strict=True)
    model.eval()
    model.to("cuda")

    dataset = pd.read_csv(args.dataset)
    entropies, snrs = dict(full=[]), dict(full=[])
    
    duration = 5.0
    num_codebook = [32, 1]
    for i, row in tqdm(dataset.iterrows(), total=len(dataset)):
        fname = os.path.basename(row.path).split('.')[0]
        # Load audio signal file
        signal = AudioSignal(row.path, duration=duration)

        # Encode audio signal as one long file
        # (may run out of GPU memory on long files)
        signal.to(model.device)

        out = model.forward(signal.audio_data, n_quantizers=num_codebook)
        y, codes = out['audio'], out['codes']
        #bands = resampler(signal.audio_data)
        entropies['codes'].append(compute_entropy(codes.squeeze(0).detach().cpu().numpy(), M=num_codebook[0], frame_rate=75))

        y, signal = y.to('cpu').detach(), signal.audio_data.to('cpu').detach()
        snrs['full'].append(ScaleInvariantSignalNoiseRatio().to("cpu")(y, signal))
        
        sf.write(
            os.path.join(args.output_path, f"{fname}_input.wav"),
            signal.reshape(-1).numpy(),
            samplerate=24_000,
        )
        sf.write(
            os.path.join(args.output_path, f"{fname}_output.wav"),
            y.reshape(-1).numpy(),
            samplerate=24_000,
        )
        
        del out
    
    for k, entropies in entropies.items():
        br_per_cb = np.array(entropies).mean(0)
        print(f'Codebook Entropy for {k}: {np.round(br_per_cb, 1)}')
        print(f'Overall Entropy for {k}: {np.round(br_per_cb.sum(), 1)}')
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
        default="/N/slate/daripete/jstsp-dac/runs/baseline2/300k/dac/weights.pth",
        required=False,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/N/slate/daripete/jstsp-dac/runs/baseline2/300k/audios",
        required=False,
    )
    args = parser.parse_args()

    main(args)
    
# latent_dim = 32
# Overall Bitrate per CB: [3976.7 4110.9 4189.7 4190. ]
# Overall SNR for midband: 2.0

# latent_dim = 64
# Overall Bitrate per CB: [3656.6 3877.5 3967.4 3877.8]
# Overall SNR for midband: 1.6

# latent_dim = 128
# Overall Bitrate per CB: [3240.9 3474.6 3558.2 3421. ]
# Overall SNR for midband: 0.5

# latent_dim = 256
# Overall Bitrate per CB: [2708.1 3011.7 3232.5 3150.4]
# Overall SNR for midband: -0.9
