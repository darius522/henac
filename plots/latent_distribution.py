import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": False,  # use mathtext instead of full LaTeX
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],  # default LaTeX-like font
    "axes.unicode_minus": False
})
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

from scipy.stats import entropy

def plot_band_latents(bands, projection='tsne'):
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })

    # Predefine pastel colors (you can add more if needed)
    pastel_colors = {
        "core": "skyblue",
        "mb": "lightcoral",
        "hb": "mediumseagreen"
    }
    band_names = {'core': 'Core', 'mb': 'MB', 'hb': 'HB'}
    priority = {"core": 3, "mb": 2, "hb": 1}
    plt.figure(figsize=(2.5, 4))

    for band_name, features in bands.items():
        features = features.detach().cpu().squeeze().numpy()
        flat_vals = features.flatten()

        # === Histogram & Entropy ===
        hist, bin_edges = np.histogram(flat_vals, bins=100, density=True)
        hist += 1e-12  # Avoid log(0) for entropy
        ent = entropy(hist, base=2)

        sns.kdeplot(
            flat_vals,
            bw_adjust=1,  # adjust bandwidth if needed
            fill=True,    # fill the area under the KDE curve
            color=pastel_colors.get(band_name, "gray"),
            label=f"{band_names[band_name]} (H={ent:.2f})",
            linewidth=1.5,  # optional: curve outline thickness
            zorder=priority[band_name]  # Set zorder based on priority
        )

    # Move legend below the plot
    plt.legend(
        title="Band Entropy",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=1,
        frameon=False
    )
    # plt.xlabel("Feature Value")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig("/N/slate/daripete/jstsp-dac/plots/band_latents_histogram_combined.pdf", dpi=150)