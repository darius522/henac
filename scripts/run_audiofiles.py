import sys
import yaml

sys.path.append("/N/slate/daripete/jstsp-dac")

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
        total_bitrate.append(codebook_entropy * frame_rate)

    return total_bitrate


def results_to_csv(fnames, entropies, snrs, path):
    data = {
        "fname": fnames,
        "entropies": entropies,
        "snrs": snrs,
    }

    df = pd.DataFrame(data)
    df.to_csv(path, index=False)

def get_chunks(signal, chunk_duration, sample_rate):
    B, C, T = signal.shape
    chunk_size = int(chunk_duration * sample_rate)  # Convert duration to samples
    
    return [signal[:, :, start : min(start + chunk_size, T)] for start in range(0, T, chunk_size)]

def get_model_args(conf_path):
    with open(conf_path, "r") as file:
        config = yaml.safe_load(file)

    # Extract DAC-specific arguments
    dac_prefix = "DAC."
    return {
        key[len(dac_prefix) :]: value
        for key, value in config.items()
        if key.startswith(dac_prefix)
    }


def main(args):

    outpath_in = os.path.join(args.model_path, "audios/input")
    outpath_out = os.path.join(args.model_path, "audios/output")
    os.makedirs(outpath_in, exist_ok=True)
    os.makedirs(outpath_out, exist_ok=True)
    conf_file = "/".join(args.model_path.split("/")[:-1]) + "/conf.yaml"

    model = dac.DAC.load(
        os.path.join(args.model_path, "dac/weights.pth"),
        strict=True,
        **get_model_args(conf_file),
    )
    model.eval()
    model.to("cuda")

    dataset = pd.read_csv(args.dataset)
    dataset = dataset.sort_values(by='path')[:1000]
    all_codes, snrs, fnames = [], [], []

    duration = 10.0
    num_codebook = [29, 1]
    for i, row in tqdm(dataset.iterrows(), total=len(dataset)):
        audio = AudioSignal(row.path, duration=30.)
        if audio.shape[-1] < duration * 24_000:
            print('Audio shorter that duration, skipping!')
            continue

        signals = get_chunks(audio, duration, 24_000)
        signals = AudioSignal.batch(signals, pad_signals=True)
        for j, signal in enumerate(signals):
            signal.to(model.device)
            fname = os.path.basename(row.path).split('.')[0] + f'_chunk_{j}'
            fnames.append(fname)

            out = model.forward(signal.audio_data, n_quantizers=num_codebook)
            y, codes = out["audio"], out["codes"]
            # bands = resampler(signal.audio_data)
            all_codes.append(codes.squeeze(0).detach().cpu().numpy())

            y, signal = y.to("cpu").detach(), signal.audio_data.to("cpu").detach()
            snrs.append(ScaleInvariantSignalNoiseRatio().to("cpu")(y, signal))

            sf.write(
                os.path.join(outpath_in, f"{fname}.wav"),
                signal.reshape(-1).numpy(),
                samplerate=24_000,
            )
            sf.write(
                os.path.join(outpath_out, f"{fname}.wav"),
                y.reshape(-1).numpy(),
                samplerate=24_000,
            )

            del out

    fr = all_codes[0].shape[-1] // duration
    codes = np.concatenate(all_codes, -1)
    bitrates = compute_entropy(codes, N=1024, M=codes.shape[0], frame_rate=fr)
    results_to_csv(
        fnames,
        np.array(bitrates).sum(), # sum over cb
        np.array(snrs),
        path=os.path.join(args.model_path, "results.csv"),
    )
    br_per_cb = np.array(bitrates)
    print(f"Codebook Entropy: {np.round(br_per_cb, 1)}")
    print(f"Overall Entropy: {np.round(br_per_cb.sum(), 1)}")
    print(f"Overall SNR: {np.round(np.mean(snrs), 1)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run compression on a dataset. Save the audio files"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/N/slate/daripete/jstsp-dac/datasets/fma_test.csv",
        required=False,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/N/slate/daripete/jstsp-dac/runs2/baseline_29cb_medium/300k",
        required=False,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
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
