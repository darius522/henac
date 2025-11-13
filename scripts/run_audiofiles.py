import sys, math
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
    outpath_out = os.path.join(args.model_path, "audios/output_tmp")
    os.makedirs(outpath_in, exist_ok=True)
    os.makedirs(outpath_out, exist_ok=True)
    conf_file = "/".join(args.model_path.split("/")[:-1]) + "/conf.yaml"

    conf_dict = get_model_args(conf_file)
    model = dac.DAC.load(
        os.path.join(args.model_path, "dac/weights.pth"),
        strict=True,
        **conf_dict,
    )
    model.eval()
    model.to("cuda")

    dataset = pd.read_csv(args.dataset)[:10]
    all_codes, snrs, fnames = [], [], []
    
    #  0: raw =    763.1, floored =        0
    #  1: raw =   1545.2, floored =     1000
    #  2: raw =   2329.1, floored =     2000
    #  3: raw =   3113.4, floored =     3000
    #  4: raw =   3899.1, floored =     3000
    #  5: raw =   4684.9, floored =     4000
    #  6: raw =   5470.8, floored =     5000
    #  7: raw =   6256.7, floored =     6000
    #  8: raw =   7044.0, floored =     7000
    #  9: raw =   7830.5, floored =     7000
    # 10: raw =   8617.2, floored =     8000
    # 11: raw =   9404.8, floored =     9000
    # 12: raw =  10192.0, floored =    10000
    # 13: raw =  10979.0, floored =    10000
    # 14: raw =  11766.0, floored =    11000
    # 15: raw =  12553.4, floored =    12000
    # 16: raw =  13340.8, floored =    13000
    # 17: raw =  14127.7, floored =    14000
    # 18: raw =  14914.6, floored =    14000
    # 19: raw =  15701.3, floored =    15000
    # 20: raw =  16488.8, floored =    16000
    # 21: raw =  17276.1, floored =    17000
    # 22: raw =  18063.7, floored =    18000
    # 23: raw =  18851.2, floored =    18000
    # 24: raw =  19639.0, floored =    19000
    # 25: raw =  20426.4, floored =    20000
    # 26: raw =  21213.8, floored =    21000
    # 27: raw =  22001.1, floored =    22000
    # 28: raw =  22788.5, floored =    22000
    # 29: raw =  23576.1, floored =    23000
    # 30: raw =  24363.5, floored =    24000

    duration = 5.0
    num_codebook = [31, 1]
    for i, row in tqdm(dataset.iterrows(), total=len(dataset)):
        audio = AudioSignal(row.path)
        if audio.shape[-1] < duration * conf_dict['sample_rate']:
            print('Audio shorter that duration, skipping!')
            continue

        signals = get_chunks(audio, duration, conf_dict['sample_rate'])
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
            y = y / torch.max(torch.abs(y)) * 0.99  # prevent clipping
            snrs.append(ScaleInvariantSignalNoiseRatio().to("cpu")(y, signal))

            sf.write(
                os.path.join(outpath_in, f"{fname}.wav"),
                signal.reshape(-1).numpy(),
                samplerate=conf_dict['sample_rate'],
            )
            sf.write(
                os.path.join(outpath_out, f"{fname}.wav"),
                y.reshape(-1).numpy(),
                samplerate=conf_dict['sample_rate'],
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
    
    new_outpath = outpath_out + f"_{int(math.floor(br_per_cb.sum() / 1000))}kbps"
    os.rename(outpath_out, new_outpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run compression on a dataset. Save the audio files"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/N/slate/daripete/jstsp-dac/datasets/fma_32khz/fma_mushra.csv",
        required=False,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/N/slate/daripete/jstsp-dac/runs_32khz/baseline_32_1cb_large_fr_80/300k",
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

