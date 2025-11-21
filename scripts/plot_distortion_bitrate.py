import os
import re
from pathlib import Path
from math import ceil
import pandas as pd
import matplotlib.pyplot as plt

def extract_bitrate(path: Path):
    """Extract integer bitrate from filename ending with '_xxxkbps.csv'."""
    m = re.search(r"_(\d+)kbps\.csv$", path.name)
    return int(m.group(1)) if m else None

def collect_means(input_dir: str):
    """
    Read CSVs and return a nested dict:
    results[metric][band] = list of (bitrate, mean_value)
    """
    input_dir = Path(input_dir)
    csv_files = list(input_dir.glob("*kbps.csv"))

    bands = ["3000", "6000", "16000", "full"]
    metrics_base = ["mel", "stft", "waveform", "sisdr"]
    full_only = ["visqol-audio"]

    # initialize
    results = {m: {b: [] for b in bands} for m in metrics_base}
    results.update({m: {"full": []} for m in full_only})  # visqol only full

    for csv_file in csv_files:
        bitrate = extract_bitrate(csv_file)
        if bitrate is None:
            continue
        df = pd.read_csv(csv_file)

        for metric in metrics_base:
            for band in bands:
                col = f"{metric}-{band}"
                if col in df.columns:
                    results[metric][band].append((bitrate, df[col].mean()))

        # full-only metrics
        for metric in full_only:
            col = f"{metric}-full"
            if col in df.columns:
                results[metric]["full"].append((bitrate, df[col].mean()))

    # sort each list by bitrate
    for metric_dict in results.values():
        for band in metric_dict:
            metric_dict[band] = sorted(metric_dict[band], key=lambda x: x[0])

    return results

def plot_metric_subplots(results, output_dir: str):
    """
    For each metric, create a figure with one subplot per band.
    Each subplot: bitrate (x) vs mean(metric) (y).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric, bands_dict in results.items():
        # only bands with data
        bands_with_data = [b for b, vals in bands_dict.items() if vals]
        if not bands_with_data:
            continue

        n_bands = len(bands_with_data)
        # 2 columns layout if multiple bands
        ncols = 2 if n_bands > 1 else 1
        nrows = ceil(n_bands / ncols)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 3.5*nrows), squeeze=False)
        axes_flat = axes.flatten()

        for ax_idx, band in enumerate(bands_with_data):
            ax = axes_flat[ax_idx]
            pairs = bands_dict[band]
            bitrates = [p[0] for p in pairs]
            means = [p[1] for p in pairs]

            ax.plot(bitrates, means, marker="o", linestyle="-")
            ax.set_title(f"{metric.upper()} — band {band} Hz")
            ax.set_xlabel("Bitrate (kbps)")
            ax.set_ylabel(f"{metric} (mean)")
            ax.grid(True, linestyle="--", alpha=0.4)

        # turn off unused subplots
        for j in range(len(bands_with_data), len(axes_flat)):
            axes_flat[j].axis("off")

        fig.suptitle(f"{metric.upper()} — Distortion vs Bitrate per Band", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        outpath = output_dir / f"distortion_bitrate_{metric}.png"
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)

def main(input_dir: str, output_dir: str):
    results = collect_means(input_dir)
    plot_metric_subplots(results, output_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot one figure per metric with one subplot per band.")
    parser.add_argument("--input_dir", required=True, help="Directory containing *_xxxkbps.csv files")
    parser.add_argument("--output_dir", required=True, help="Directory to save output plots")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
