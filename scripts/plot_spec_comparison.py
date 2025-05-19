import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# === USER CONFIGURATION ===
ROOT_DIR = '/N/slate/daripete/jstsp-dac/datasets/mushra'  # <- Set your root folder here
OUTPUT_DIR = '/N/slate/daripete/jstsp-dac/plots/appendix'
SR = 24000
MODELS = ['gt', 'a', 'std', 'b', 'c']  # Adjust as needed
PRETTY_MODELS = {'a':r"DAC$_{(29)}$",
                'b':r"HE-NAC$_{(24, 2)}$",
                'c':r"HE-NAC$_{(18, 2, 1)}$",
                'std':r"HE-AAC",
                'gt':r"Reference"}
FREQS = [[6000,12000]]
N_FFT = 1024
HOP_LENGTH = 256
CMAP = 'inferno'
DPI = 400
EPS = 1e-10  # Small value to avoid log(0)
DB_MIN = -60
DB_MAX = 0

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

# Expand path and add all fonts in ~/.fonts to matplotlib
# font_dir = os.path.expanduser("~/.fonts")
# font_files = font_manager.findSystemFonts(fontpaths=[font_dir])

# for font_file in font_files:
#     print(font_file)
font_manager.fontManager.addfont("/N/slate/daripete/jstsp-dac/scripts/cmu.serif-roman.ttf")

# OPTIONAL: Set a specific font family if desired
font_prop = font_manager.FontProperties(fname="/N/slate/daripete/jstsp-dac/scripts/cmu.serif-roman.ttf")
rcParams["font.family"] = font_prop.get_name()

# Set mathtext to use Computer Modern (matches LaTeX default math fonts)
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["mathtext.rm"] = font_prop.get_name()
plt.rcParams["font.size"] = 24
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_common_filenames():
    ref_folder = os.path.join(ROOT_DIR, MODELS[0])
    filenames = [f for f in os.listdir(ref_folder) if f.endswith('.wav')]
    return sorted([f[len(MODELS[0]) + 1:] for f in filenames if f.startswith(MODELS[0] + "_")])

def compute_log_spectrogram(y, fmin, fmax):
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=N_FFT))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    frequencies = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)

    # Find index range for 6–12 kHz
    freq_mask = (frequencies >= fmin) & (frequencies <= fmax)
    S_db_cropped = S_db[freq_mask, :]
    return S_db_cropped

def plot_spectrogram_row(spectrograms, output_path, fmin=0, fmax=SR // 2):
    num_models = len(spectrograms)
    fig_width = num_models * 8  # Adjust per model
    fig, axes = plt.subplots(1, num_models, figsize=(fig_width, 5), dpi=DPI, sharey=True)

    if num_models == 1:
        axes = [axes]
    
    for idx, (ax, S, M) in enumerate(zip(axes, spectrograms, MODELS)):
        librosa.display.specshow(S, sr=SR, hop_length=HOP_LENGTH, x_axis='time', y_axis='linear', cmap=CMAP, ax=ax)
        # ax.axis('off')
        ax.set_title(PRETTY_MODELS[M])  # Remove title for each subplot

        if idx == 0:
            # Get y-axis limits
            y_min, y_max = ax.get_ylim()

            # Turn off only what you need to, keep ticks/spine
            ax.tick_params(
                axis='y',
                direction='out',
                length=4,
                width=1,
                labelsize=24
            )

            # Set only bottom and top ticks
            ax.set_yticks([y_min, y_max])
            ax.set_yticklabels([f"{fmin} kHz", f"{fmax} kHz"])

            # Keep only left spine
            ax.spines['left'].set_visible(True)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

            # Only show ticks on left
            ax.yaxis.set_ticks_position('left')

        else:
            # Fully clean up all other subplots
            ax.axis('off')

    plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def main():
    common_filenames = get_common_filenames()
    
    for common_name in common_filenames:
        spectrograms = []
        for freqs in FREQS:
            plt.cla()
            plt.clf()
            for model in MODELS:
                full_filename = f"{model}_{common_name}"
                filepath = os.path.join(ROOT_DIR, model, full_filename)
                y, sr = sf.read(filepath, start=0, stop=None)
                if sr != SR:
                    y = librosa.resample(y, orig_sr=sr, target_sr=SR)
                if y.ndim > 1:
                    y = y.mean(axis=1)
                S_log = compute_log_spectrogram(y, fmin=freqs[0], fmax=freqs[1])
                spectrograms.append(S_log)
            
            out_path = os.path.join(OUTPUT_DIR, f"{common_name.replace('.wav', '')}.png")
            plot_spectrogram_row(spectrograms, out_path, fmin=freqs[0], fmax=freqs[1])
            print(f"Saved: {out_path}")

if __name__ == '__main__':
    main()
