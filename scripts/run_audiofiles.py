import sys, math, ast
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

import argparse, os, shutil

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

    return [
        signal[:, :, start : min(start + chunk_size, T)]
        for start in range(0, T, chunk_size)
    ]
    
def match_rms(input_sig, output_sig):
    in_rms = np.sqrt(np.mean(input_sig**2))
    out_rms = np.sqrt(np.mean(output_sig**2))
    if out_rms == 0:
        return output_sig
    scale = in_rms / out_rms
    return output_sig * scale

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

    num_codebook = ast.literal_eval(args.num_codebooks)

    dataset = pd.read_csv(args.dataset)[:1000]
    all_codes, snrs, fnames = [], [], []

    outpath = args.output_path if args.output_path is not None else args.model_path
    outpath_in = os.path.join(outpath, "input")
    outpath_out = os.path.join(
        outpath, f"output_{num_codebook}"
    )
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

    duration = 10.0

    for i, row in tqdm(dataset.iterrows(), total=len(dataset)):
        audio = AudioSignal(row.path)
        if audio.shape[-1] < duration * conf_dict["sample_rate"]:
            print("Audio shorter that duration, skipping!")
            continue

        signals = get_chunks(audio, duration, conf_dict["sample_rate"])
        signals = AudioSignal.batch(signals, pad_signals=True)
        for j, signal in enumerate(signals):
            try:
                signal.to(model.device)
                fname = os.path.basename(row.path).split(".")[0] + f"_chunk_{j}"
                fnames.append(fname)

                out = model.forward(signal.audio_data, n_quantizers=num_codebook)
                y, codes = out["audio"], out["codes"]
                # bands = resampler(signal.audio_data)
                all_codes.append(codes.squeeze(0).detach().cpu().numpy())

                y, signal = y.to("cpu").detach(), signal.audio_data.to("cpu").detach()
                y = y / torch.max(torch.abs(y)) * 0.99  # prevent clipping
                snrs.append(ScaleInvariantSignalNoiseRatio().to("cpu")(y, signal))
                
                input_sig = signal.reshape(-1).numpy()
                output_sig = y.reshape(-1).numpy()
                output_sig = match_rms(input_sig, output_sig)

                # sf.write(
                #     os.path.join(outpath_in, f"{fname}.wav"),
                #     input_sig,
                #     samplerate=conf_dict["sample_rate"],
                # )
                sf.write(
                    os.path.join(outpath_out, f"{fname}.wav"),
                    output_sig,
                    samplerate=conf_dict["sample_rate"],
                )

                del out
            except Exception as e:
                print(f"Error processing {row.path} chunk {j}: {e}")
                continue

    fr = all_codes[0].shape[-1] // duration
    codes = np.concatenate(all_codes, -1)
    bitrates = compute_entropy(codes, N=1024, M=codes.shape[0], frame_rate=fr)
    results_to_csv(
        fnames,
        np.array(bitrates).sum(),  # sum over cb
        np.array(snrs),
        path=os.path.join(args.model_path, "results.csv"),
    )
    br_per_cb = np.array(bitrates)
    print(f"Codebook Entropy: {np.round(br_per_cb, 1)}")
    print(f"Overall Entropy: {np.round(br_per_cb.sum(), 1)}")
    print(f"Overall SNR: {np.round(np.mean(snrs), 1)}")

    new_outpath = outpath_out + f"_{int(round(br_per_cb.sum() / 1000, 0))}kbps"
    # If destination exists, delete it
    if os.path.exists(new_outpath):
        shutil.rmtree(new_outpath)
    os.rename(outpath_out, new_outpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run compression on a dataset. Save the audio files"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/N/slate/daripete/jstsp-dac/datasets/fma_24khz/fma_mushra.csv",
        required=False,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/N/slate/daripete/jstsp-dac/runs_24khz_bis/baseline_32_1cb_large_fr_75/300k",
        required=False,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/N/slate/daripete/jstsp-dac/datasets/mushra_24khz",
        required=False,
    )
    parser.add_argument('--num_codebooks', type=str, required=True, 
                    help='List of integers, e.g. "[31, 1]"')
    args = parser.parse_args()

    main(args)
