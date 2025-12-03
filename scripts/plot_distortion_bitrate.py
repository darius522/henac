import os
import re
from pathlib import Path
from math import ceil
import pandas as pd

import matplotlib
font = {'family' : 'normal',
    'size'   : 15}
matplotlib.rc('font', **font)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

def extract_bitrate(path: Path):
    m = re.search(r"_(\d+)kbps\.csv$", path.name)
    return int(m.group(1)) if m else None

def collect_means(input_dir: str):
    input_dir = Path(input_dir)
    csv_files = list(input_dir.glob("*kbps.csv"))

    bands = ["3000", "6000", "16000", "full"]
    metrics_base = ["mel", "stft", "waveform", "sisdr"]
    full_only = ["visqol-audio"]

    results = {m: {b: [] for b in bands} for m in metrics_base}
    results.update({m: {"full": []} for m in full_only})

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

        for metric in full_only:
            col = f"{metric}-full"
            if col in df.columns:
                results[metric]["full"].append((bitrate, df[col].mean()))

    for metric_dict in results.values():
        for band in metric_dict:
            metric_dict[band] = sorted(metric_dict[band], key=lambda x: x[0])

    return results


# -------------------------------------------------------------------------
# Collect two systems' results
# -------------------------------------------------------------------------
def collect_multiple(dir1: str, dir2: str):
    return {
        "system1": collect_means(dir1),
        "system2": collect_means(dir2),
    }


# -------------------------------------------------------------------------
# UPDATED: now optionally takes a `codebooks` dict
# -------------------------------------------------------------------------
def offset_for_point(x_vals, y_vals, i, base=0.015):
    """Return a small y-offset based on slope direction."""
    if i == 0:
        slope = y_vals[1] - y_vals[0]
    elif i == len(y_vals) - 1:
        slope = y_vals[-1] - y_vals[-2]
    else:
        slope = y_vals[i+1] - y_vals[i-1]

    # If curve going down → place label slightly above, else below.
    return base if slope < 0 else -base

def plot_metric_subplots(all_results, output_dir: str):
    """
    all_results = {
        "system1": results1,
        "system2": results2,
        "codebooks": optional dict with:
            {
                "system1": { band: {bitrate: codebook, ...}, ... },
                "system2": { band: {bitrate: codebook, ...}, ... }
            }
    }
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = all_results["system1"].keys()

    system_styles = {
        "system1": {"marker": "o", "linestyle": "-", "color": "black", "label": "DAC"},
        "system2": {"marker": "s", "linestyle": "--", "color": "tab:red", "label": "HE-NAC"},
    }

    has_codebooks = "codebooks" in all_results

    for metric in metrics:
        bands_dict_sys1 = all_results["system1"][metric]
        bands_with_data = [b for b, vals in bands_dict_sys1.items() if vals]
        if not bands_with_data:
            continue

        n_bands = len(bands_with_data)
        ncols = 2 if n_bands > 1 else 1
        nrows = ceil(n_bands / ncols)

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(8 * ncols, 4.5 * nrows),
            squeeze=False,
            sharex=True
        )
        axes_flat = axes.flatten()

        for ax_idx, band in enumerate(bands_with_data):
            ax = axes_flat[ax_idx]

            # --- plot each system's curve ---
            for system_name, result_dict in all_results.items():
                if system_name == "codebooks":
                    continue

                pairs = result_dict[metric][band]
                if not pairs:
                    continue

                bitrates = [p[0] for p in pairs]

                if metric == "sisdr":
                    means = [np.floor(p[1]) for p in pairs]
                else:
                    means = [np.floor(p[1] * 100.) / 100. for p in pairs]

                style = system_styles[system_name]

                ax.plot(
                    bitrates,
                    means,
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    label=style["label"],
                    color=style["color"],
                )

                # -----------------------------------------------------
                # INLINE CODEBOOK LABELS
                # -----------------------------------------------------
                if has_codebooks and system_name in all_results["codebooks"]:
                    if band in all_results["codebooks"][system_name]:
                        cb_map = all_results["codebooks"][system_name][band]

                        for i, (x, y) in enumerate(zip(bitrates, means)):
                            if x in cb_map:
                                cb = cb_map[x]
                                y_offset = offset_for_point(bitrates, means, i) * (ax.get_ylim()[1] - ax.get_ylim()[0])
                                
                                dy = 6
                                if ax_idx == 2 and system_name == "system2" and i < 3:
                                    dy = -12  # pixels down

                                ax.annotate(
                                    f"c{cb}",
                                    (x, y),
                                    textcoords="offset points",
                                    xytext=(0, dy),   # 6 pixels upward
                                    ha="center",
                                    fontsize=10,
                                    color=style["color"],
                                    alpha=0.8,
                                    clip_on=True,
                                )

            acr, prev_band = ("CB", "0") if band == "3000" else ("MB", "3") if band == "6000" else ("HB", "6") if band == "16000" else ("Fullband", "0")
            bandrange = f"{prev_band} - {int(band)//1000} kHz" if band != "full" else f"{prev_band} - {16} kHz"

            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax + 0.05 * (ymax - ymin))
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
            ax.set_title(f"{metric.upper()} " + r"$L_1$" + f" Error — {bandrange} ({acr})")
            
            if ax_idx >= 2:
                ax.set_xlabel("Bitrate (kbps)")
            if ax_idx % nrows == 0:
                ax.set_ylabel(f"{metric} error (mean)")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend()

        # turn off unused subplots
        for j in range(len(bands_with_data), len(axes_flat)):
            axes_flat[j].axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        fig.savefig(output_dir / f"{metric}_comparison.png", dpi=200, bbox_inches="tight")
        fig.savefig(output_dir / f"{metric}_comparison.pdf", dpi=200, bbox_inches="tight")
        plt.close(fig)


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
def main(input_dir1: str, input_dir2: str, output_dir: str, codebooks=None):
    all_results = collect_multiple(input_dir1, input_dir2)
    if codebooks is not None:
        all_results["codebooks"] = codebooks
    plot_metric_subplots(all_results, output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare two systems' distortion-vs-bitrate curves.")
    parser.add_argument("--input_dir1", required=True)
    parser.add_argument("--input_dir2", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    # If needed, you can load or define codebooks here
    codebooks = {
        "system1": {
            "3000": {14: 18, 16: 21, 17: 22, 19: 25, 21: 27, 24: 31},
            "6000": {14: 18, 16: 21, 17: 22, 19: 25, 21: 27, 24: 31},
            "16000": {14: 18, 16: 21, 17: 22, 19: 25, 21: 27, 24: 31},
        },
        "system2": {
            "3000": {14: 16, 16: 16, 17: 16, 19: 16, 21: 16, 24: 16},
            "6000": {14: 1, 16: 2, 17: 3, 19: 4, 21: 4, 24: 4},
            "16000": {14: 0, 16: 0, 17: 0, 19: 0, 21: 1, 24: 2},
        },
    }
    main(args.input_dir1, args.input_dir2, args.output_dir, codebooks=codebooks)
