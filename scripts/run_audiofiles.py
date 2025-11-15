import sys, math

sys.path.append("/N/slate/daripete/jstsp-dac")

import dac
from audiotools import AudioSignal

import yaml
import pandas as pd
from tqdm import tqdm
import soundfile as sf
import numpy as np
import torch
from matplotlib import pyplot as plt
import librosa

from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

import argparse, os

from utils.audio_utils import normalize_to_match_peak_batched

import julius

import gc

from utils.latent_distribution import plot_band_latents, plot_codebook_indices


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


def spectrogram(signal, path):
    plt.clf()
    plt.cla()
    f = plt.figure(figsize=(10, 4))
    D = np.abs(librosa.stft(signal.audio_data.squeeze().numpy()))
    librosa.display.specshow(
        librosa.amplitude_to_db(D, ref=np.max), sr=32_000, hop_length=512, cmap="magma"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    del f



def results_to_csv(fnames, entropies, snrs, path):
    # Convert each entry to numpy arrays and sum along the last axis
    entropies = {k: np.array(v).sum(-1) for k, v in entropies.items()}

    # Available bands (e.g., could be ["core"], ["core", "mb"], ["core", "mb", "hb"])
    bands = ["core", "mb", "hb"]
    available = [b for b in bands if b in entropies]

    # Compute total entropy as the sum of all available bands
    entropies_all = sum(entropies[b] for b in available)

    # Build the data dict dynamically (only for available bands)
    data = {"fname": fnames, "entropies_all": entropies_all}
    for b in available:
        data[f"entropies_{b}"] = entropies[b]
        data[f"snrs_{b}"] = snrs.get(b, np.zeros_like(entropies[b]))

    # Create and save the DataFrame
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)


def get_chunks(signal, chunk_duration, sample_rate):
    B, C, T = signal.shape
    chunk_size = int(chunk_duration * sample_rate)  # Convert duration to samples

    return [
        signal[:, :, start : min(start + chunk_size, T)]
        for start in range(0, T, chunk_size)
    ]


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

    dataset = pd.read_csv(args.dataset)[:10]
    num_codebooks = [[16, 4, 2]]
    # num_codebooks = [
    #     # [[True, False], [True, False]], # 18, 0, 0
    #     # [[True, True], [True, False]], # 18, 2, 0
    #     # [[True, True], [True, True]], # 18, 2, 2
    #     # [[False, True], [True, True]], # 0, 2, 2
    #     [[True, True], [True, True]],  # 0, 0, 2
    #     # [[True, False], [False, True]], # 18, 0, 2
    # ]
    for num_codebook in num_codebooks:
        outpath_in = os.path.join(
            args.model_path, f"audios/input"
        )  # _{num_codebook}")
        outpath_out = os.path.join(
            args.model_path, f"audios/output"
        )  # _{num_codebook}")
        os.makedirs(outpath_in, exist_ok=True)
        os.makedirs(outpath_out, exist_ok=True)
        conf_file = "/".join(args.model_path.split("/")[:-1]) + "/conf.yaml"
        conf = get_model_args(conf_file)

        model = dac.DAC.load(
            os.path.join(args.model_path, "dac/weights.pth"), strict=True, **conf
        )
        model.eval()
        model.to("cuda")

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters())

        # Or to count only trainable parameters:
        def count_trainable_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Example usage:
        print("Total parameters:", count_parameters(model) / 1e6)
        print("Trainable parameters:", count_trainable_parameters(model) / 1e6)

        all_feats, all_codes, snrs, fnames = (
            dict(core=[], mb=[], hb=[]),
            dict(core=[], mb=[], hb=[]),
            dict(core=[], mb=[], hb=[]),
            [],
        )

        # Calculate cutoffs based on num_codebook per bands
        _, mb, hb = num_codebook

        if mb != 0 and hb != 0:
            cutoffs = [3000, 6000]
        elif mb != 0 and hb == 0:
            cutoffs = [3000]
        elif mb == 0 and hb == 0:
            cutoffs = [int(conf["sample_rate"]//2)]
        else:
            cutoffs = []

        resampler = julius.SplitBands(conf["sample_rate"], cutoffs=cutoffs).to("cuda")
        duration = 5.0
        for i, row in tqdm(dataset.iterrows(), total=len(dataset)):
            audio = AudioSignal(row.path)
            if audio.shape[-1] < duration * conf["sample_rate"]:
                print("Audio shorter that duration, skipping!")
                continue

            signals = get_chunks(audio, duration, conf["sample_rate"])
            signals = AudioSignal.batch(signals, pad_signals=True)
            for j, signal in enumerate(signals):
                signal.to(model.device)
                fname = os.path.basename(row.path).split(".")[0] + f"_chunk_{j}"
                fnames.append(fname)

                with torch.no_grad():
                    non_zero_cbs = [
                        conf["n_codebooks"],
                        conf["skip_args"][0]["n_codebooks"],
                        conf["skip_args"][1]["n_codebooks"],
                    ]
                    non_zero_cbs = [
                        a if a != 0 else b for a, b in zip(num_codebook, non_zero_cbs)
                    ]
                    out = model.infer_bands(signal.audio_data, n_quantizers=non_zero_cbs)
                    y_bands, codes, feats = out["audio"], out["codes"], out["feats"]
                    keys = list(y_bands.keys())

                    # mask for which bands to keep
                    keep_mask = [n != 0 for n in num_codebook]

                    # filter each dict accordingly
                    y_bands = {
                        k: v
                        for k, v, keep in zip(keys, y_bands.values(), keep_mask)
                        if keep
                    }
                    codes = {
                        k: v for k, v, keep in zip(keys, codes.values(), keep_mask) if keep
                    }
                    feats = {
                        k: v for k, v, keep in zip(keys, feats.values(), keep_mask) if keep
                    }

                for k, c in codes.items():  # k, [B, CB, T]
                    nc, fr = c.shape[1], int(c.shape[-1] // duration)
                    all_codes[k].append(c.squeeze(0).detach().cpu().numpy())

                for k, c in feats.items():  # k, [B, CB, T]
                    all_feats[k].append(c.squeeze(0).detach().cpu().numpy())

                signal_bands = resampler(signal.audio_data)

                mb_rec = np.zeros(signal.audio_data.shape)
                for i, ((k, y_band), signal_band) in enumerate(
                    zip(y_bands.items(), signal_bands)
                ):
                    y_band = resampler(y_band)[i]
                    y_band = y_band.reshape(1, 1, -1).to("cuda")
                    signal_band = signal_band.reshape(1, 1, -1).to("cuda")
                    snrs[k].append(
                        ScaleInvariantSignalNoiseRatio()
                        .to("cuda")(y_band, signal_band)
                        .detach()
                        .cpu()
                        .item()
                    )

                    mb_rec = mb_rec + y_band.cpu().detach().numpy()
                    del y_band

                # for k, y_band in y_bands.items():
                #     sf.write(
                #         os.path.join(args.output_path, f"{fname}_{k}.wav"),
                #         y_band.reshape(-1).cpu().detach().numpy(),
                #         samplerate=32_000,
                #     )
                sf.write(
                    os.path.join(outpath_in, f"{fname}.wav"),
                    signal.audio_data.reshape(-1).cpu().detach().numpy(),
                    samplerate=conf["sample_rate"],
                )
                sf.write(
                    os.path.join(outpath_out, f"{fname}.wav"),
                    mb_rec.reshape(-1),
                    samplerate=conf["sample_rate"],
                )

                for k in list(codes.keys()):  # Convert to list to avoid runtime errors
                    codes[k] = codes[k].detach().cpu()
                del codes

                for k in list(y_bands.keys()):
                    y_bands[k] = y_bands[k].detach().cpu()
                del y_bands
                torch.cuda.empty_cache()
                gc.collect()

        for k, v in all_feats.items():
            if len(v) > 0:
                all_feats[k] = np.concatenate(v, -1)

        # plot_codebook_indices({k: np.concatenate(v, -1) for k, v in all_codes.items()})
        # plot_band_latents(feats, projection='umap')

        bitrates = dict()
        for k, v in all_codes.items():
            if len(v) > 0:
                fr = v[0].shape[-1] // duration
                codes = np.concatenate(v, -1)
                bitrates[k] = compute_entropy(codes, N=1024, M=codes.shape[0], frame_rate=fr)

        results_to_csv(
            fnames, bitrates, snrs, path=os.path.join(args.model_path, "results.csv")
        )
        tot_ent = 0.0
        for k, bitrates in bitrates.items():
            br_per_cb = np.array(bitrates)
            print(f"Codebook Entropy for {k}: {np.round(br_per_cb, 1)}")
            print(f"Overall Entropy for {k}: {np.round(br_per_cb.sum())}")
            tot_ent += br_per_cb.sum()
        print(f"Overall Total Entropy: {np.round(tot_ent, 1)}")
        for k, v in snrs.items():
            print(f"Overall SNR for {k}: {np.round(np.mean(v), 1)}")
            
        # lastly rename output folder to include bitrate info

        new_outpath = outpath_out + f"_{int(math.floor(tot_ent / 1000))}kbps"
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
        default="/N/slate/daripete/jstsp-dac/runs_32khz/hb_32_1cb_4_1cb_2_1cb_fr_80_320_500/300k",
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
