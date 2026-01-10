import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from tqdm import tqdm

CLIP_AMP = -35.

import matplotlib
font = {'family' : 'normal',
    'size'   : 23}
matplotlib.rc('font', **font)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['axes.unicode_minus'] = False

# Adobe-style colormap
adobe_colors = [
    (0.00, "#000000"),
    (0.10, "#1A0033"),
    (0.25, "#4B0066"),
    (0.45, "#990033"),
    (0.65, "#FF3300"),
    (0.80, "#FF9900"),
    (0.93, "#FFCC00"),
    (1.00, "#FFFFFF"),
]
cmap_adobe = LinearSegmentedColormap.from_list("adobe", adobe_colors)

def compute_spectrogram(path, max_seconds=5):
    y, sr = librosa.load(path, sr=None)
    y = y[:sr * max_seconds]

    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=64))
    S_db = librosa.amplitude_to_db(S)

    # Adobe-style processing
    S_db_clip = np.clip(S_db, CLIP_AMP, None)
    S_norm = (S_db_clip - S_db_clip.min()) / (S_db_clip.max() - S_db_clip.min())
    gamma = 0.65
    S_gamma = S_norm ** gamma
    contrast = 1.35
    S_vis = S_gamma ** (1 / contrast)

    return S_vis, sr


def compare_directories_multirow_dict(dir_dict, output_path, zoom_freq=(6000, 16000), max_files=None, fnames=None):
    """
    Create a single multi-row, multi-column figure comparing codecs.
    
    dir_dict: dict of {system_name: directory_path}
    output_path: path to save the combined figure
    zoom_freq: frequency range to zoom (Hz)
    max_files: optionally limit the number of rows for clarity
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    system_names = list(dir_dict.keys())
    dir_list = list(dir_dict.values())

    # Find common files
    file_sets = [set(os.listdir(d)) for d in dir_list]
    common_files = sorted(set.intersection(*file_sets))
    common_files = [f for f in common_files if f in fnames]  # filter to specific files
    if max_files:
        common_files = common_files[:max_files]

    print(f"Found {len(common_files)} shared files.")

    n_rows = len(common_files)
    n_cols = len(dir_list)

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        dpi=200,
        sharey=True,   # share y-axis across columns
        constrained_layout=False)
    
    fig.subplots_adjust(
        wspace=0.05,   # horizontal spacing between columns
        hspace=0.05    # vertical spacing between rows
    )

    # Ensure axes is 2D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    elif n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    # Define y-axis ticks (in Hz)
    y_ticks = [zoom_freq[0], zoom_freq[1]]
    y_labels = [f"{t/1000:.1f} kHz" for t in y_ticks]  # convert to kHz

    for row_idx, fname in enumerate(tqdm(common_files, desc="Processing files")):
        for col_idx, d in enumerate(dir_list):
            path = os.path.join(d, fname)
            S_vis, sr = compute_spectrogram(path)

            ax = axes[row_idx, col_idx]
            librosa.display.specshow(
                S_vis,
                sr=sr,
                x_axis=None,  # hide x-axis
                y_axis='log',
                cmap=cmap_adobe,
                ax=ax
            )

            # ax.set_yscale("log")
            ax.set_ylim(zoom_freq[0], zoom_freq[1])

            ax.set_ylabel("")
            if col_idx == 0:
                ax.text(
                    -0.05, 0.02, "3 kHz",        # (x, y) in axes coords
                    transform=ax.transAxes,
                    va="bottom",
                    ha="right",
                )

                # Top label
                ax.text(
                    -0.05, 0.98, "16 kHz",
                    transform=ax.transAxes,
                    va="top",
                    ha="right",
    )
            else:
                # Other columns: hide y-axis
                ax.set_yticks([])
                ax.set_yticklabels([])

            if row_idx == 0:
                ax.set_title(system_names[col_idx], pad=12)

    # plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined figure to {output_path}")


if __name__ == "__main__":
    ablations_type = 1  # Change this to select different ablation sets
    if ablations_type == 0:
        directories = {
            "Input": "/N/slate/daripete/jstsp-dac/datasets/ablations_32khz/input_ablations",
            r"HE-NAC$_{(16,0,0)}$": "/N/slate/daripete/jstsp-dac/datasets/ablations_32khz/henac_ablations_output_[16, 0, 0]",
            r"HE-NAC$_{(16,4,0)}$": "/N/slate/daripete/jstsp-dac/datasets/ablations_32khz/henac_ablations_output_[16, 4, 0]",
            r"HE-NAC$_{(16,4,2)}$": "/N/slate/daripete/jstsp-dac/datasets/ablations_32khz/henac_ablations_output_[16, 4, 2]",
        }
        fnames = [
            "013164_chunk_4.wav",
            "001662_chunk_2.wav",
            "023569_chunk_5.wav",
            "028045_chunk_0.wav",
            "031474_chunk_0.wav",
            "060374_chunk_0.wav",
            "102171_chunk_2.wav",
        ]
    elif ablations_type == 1:
        directories = {
            "Input": "/N/slate/daripete/jstsp-dac/datasets/ablations_32khz/input_ablations",
            r"DAC$_{(31)}$": "/N/slate/daripete/jstsp-dac/datasets/ablations_32khz/dac_ablations_output_[31, 1]",
            r"HE-AAC v1$": "/N/slate/daripete/jstsp-dac/datasets/ablations_32khz/heaac_ablations",
            r"HE-NAC$_{(16,4,2)}$": "/N/slate/daripete/jstsp-dac/datasets/ablations_32khz/henac_ablations_output_[16, 4, 2]",
        }
        fnames = [
            "026699_chunk_0.wav",
            "037246_chunk_0.wav",
            "045489_chunk_5.wav",
            "060374_chunk_2.wav",
            "119193_chunk_0.wav",
            "120128_chunk_1.wav",
            "136177_chunk_4.wav",
        ]


    compare_directories_multirow_dict(
        dir_dict=directories,
        output_path=f"/N/slate/daripete/jstsp-dac/plots/appendix_ablations_32khz_combined_{ablations_type}.pdf",
        zoom_freq=(3000, 16000),
        max_files=15,
        fnames=fnames
    )