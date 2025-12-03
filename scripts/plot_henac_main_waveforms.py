import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf  # `pip install soundfile`
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

def plot_toy_waveform(
    audio_path,
    ax=None,
    fade_pct=1.0,        # percentage fade-in/out
    downsample=200,     # toy-ish: fewer points = simple waveform
    linewidth=3,
    color="black"
):
    # Load audio
    y, sr = sf.read(audio_path)
    y = (y + np.flipud(y)) / 2
    y = y / np.max(np.abs(y))
    
    # Mono for simplicity
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # ---- Step 1: Downsample strongly ----
    idx = np.linspace(0, len(y)-1, downsample).astype(int)
    y_small = y[idx]
    x_small = np.linspace(0, 1, downsample)

    # ---- Step 2: Cubic spline interpolation ----
    interp = interp1d(x_small, y_small, kind='cubic')
    x_fine = np.linspace(0, 1, 2000)
    y_smooth = interp(x_fine)

    # ---- Step 3 (optional): gentle smoothing ----
    y = gaussian_filter1d(y_smooth, sigma=1.5)

    # Apply fade-in/out
    n = len(y)
    fade_len = int(n * fade_pct)
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)

    y[:fade_len] *= fade_in
    y[-fade_len:] *= fade_out

    # Use existing axis or create one
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 2))

    # Plot toy waveform
    ax.plot(y, linewidth=linewidth,
            solid_joinstyle='round',
            solid_capstyle='round', 
            color=color)
    # ax.fill_between(np.linspace(0, 1, len(y)), y, 0, alpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([0, len(y)])
    ax.set_ylim([np.min(y)*1.05, np.max(y)*1.05])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    return ax

band_cfg = {'cb': {'color': '#BD4B4B', 'linewidth': 8., 'downsample': 50}, 
            'mb': {'color': '#567199', 'linewidth': 4., 'downsample': 100}, 
            'hb': {'color': '#6B9455', 'linewidth': 2., 'downsample': 1000}}

for band, cfg in band_cfg.items():
    fig, ax = plt.subplots(figsize=(4,2))
    plot_toy_waveform("/N/slate/daripete/jstsp-dac/runs_32khz/baseline_32_1cb_xlarge_fr_80/300k/audios/input/000852_chunk_1.wav", ax=ax, **cfg)
    plt.savefig(f"./{band}.png", bbox_inches='tight', dpi=300, transparent=True)

# fig, ax = plt.subplots(figsize=(6,2))
# for band, cfg in band_cfg.items():
#     plot_toy_waveform("/N/slate/daripete/jstsp-dac/runs_32khz/baseline_32_1cb_xlarge_fr_80/300k/audios/input/000852_chunk_1.wav", ax=ax, **cfg)
# plt.savefig(f"./full.png", bbox_inches='tight', dpi=300, transparent=True)
