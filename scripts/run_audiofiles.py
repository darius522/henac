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
import matplotlib.pyplot as plt

from utils.audio_utils import resample_bands
from julius import resample_frac

from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

import argparse

ROOT = os.getcwd()

DURATION = 3.

def indices_to_entropy(indices, time_axis=1, eps=1e-20, size=1024) -> torch.Tensor:
    n_step = indices.shape[time_axis]
    oh_indices = torch.nn.functional.one_hot(indices, num_classes=size)
    p = (torch.sum(oh_indices, dim=time_axis) + eps) / n_step
    return -torch.sum(torch.mul(p, torch.log(p)), axis=-1)  # * n_step

def draw_rvq_histogram(data,save_path):
    assert len(data) == 32
    plt.clf()
    plt.cla()
    # Create a figure with subplots
    fig, axs = plt.subplots(8, 4, figsize=(12, 18))  # 8 rows, 4 columns

    # Flatten the axes array for easy iteration
    axs = axs.flatten()

    # Loop through each row and plot the histogram
    for i in range(32):
        axs[i].hist(data[i], bins=32, color='blue', alpha=0.7)
        axs[i].set_title(f'RVQ#{i+1}')
        axs[i].set_xlabel('Value')
        axs[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(save_path)


def main(args):
    if not os.path.isdir(os.path.join(args.output_path, 'audios')):
        os.makedirs(os.path.join(args.output_path, 'audios'))

    model = dac.DAC.load(args.model_path)
    model.to("cuda:3")
    model.eval()

    dataset = pd.read_csv(args.dataset)
    bitrates, snrs = {}, []

    for i, row in tqdm(dataset.iterrows(), total=len(dataset)):
        fname = os.path.basename(row.path).split(".")[0]
        # Load audio signal file
        signal = AudioSignal(row.path, duration=DURATION)

        # Encode audio signal as one long file
        # (may run out of GPU memory on long files)
        signal.to(model.device)

        x = model.preprocess(signal.audio_data, signal.sample_rate)
        z, cb_codes, _, _, _, skips = model.encode(x)

        # Decode audio signal
        y: AudioSignal
        skips = model.autoencode_skips(skips, return_output=True, n_quantizers=None)
        y = model.multidecode(z, [s['audio'] for s in skips])
        y = y[:2]
        mix = torch.stack(y).sum(0)

        mix, signal = mix.to("cpu").detach(), signal.audio_data.to("cpu").detach()
        snrs.append(ScaleInvariantSignalNoiseRatio().to("cpu")(mix, signal))

        sf.write(
            os.path.join(args.output_path, 'audios', f"{fname}_full_i.wav"),
            signal.reshape(-1).numpy(),
            samplerate=24_000,
        )
        for j, yy in enumerate(y):
            sf.write(
                os.path.join(args.output_path, 'audios', f"{fname}_full_o_{j}.wav"),
                yy.detach().cpu().reshape(-1).numpy(),
                samplerate=24_000,
            )

        # code: torch.Tensor
        # for code, title in zip([cb_codes, *[s['codes'] for s in skips]], ['CB', 'MB', 'HB']):
        #     entropy = indices_to_entropy(code.permute(0, 2, 1), time_axis=1, size=1024).sum().cpu().item()
        #     bitrates[f"{title}_{code.shape[-1]//DURATION}"] = entropy * (code.shape[-1] // DURATION)
            # draw_rvq_histogram(
            #     code.squeeze(0).cpu().detach().numpy(),
            #     os.path.join(args.output_path, "plots", title + ".png"),
            # )

    print(f"Per-Code Bitrate")
    for k, v in bitrates.items():
        print(f"Code {k}: {np.round(np.mean(v), 2)}")
    print(f"Overall SNR: {np.round(np.mean(snrs), 1)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run compression on a dataset. Save the audio files"
    )
    parser.add_argument(
        "--dataset",
        type=str, 
        default=f"{ROOT}/datasets/fma_test_subset.csv",
        required=False,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=f"{ROOT}/runs/stage1_1skip_dim_8_nodropout_noadv2/best/dac/weights.pth",
        required=False,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=f"{ROOT}/runs/stage1_1skip_dim_8_nodropout_noadv2",
        required=False,
    )
    args = parser.parse_args()

    main(args)
