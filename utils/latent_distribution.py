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


def plot_codebook_indices(data):
    for band, tensor in data.items():
        fig, axes = plt.subplots(1, tensor.shape[0], figsize=(3*tensor.shape[0], 3))
        if tensor.shape[0] == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            ax.hist(tensor[i], bins=np.arange(tensor[i].max()+2)-0.5)
            ax.set_title(f"{band} CB{i}")
            ax.set_xlabel("Index")
            ax.set_ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"/N/slate/daripete/jstsp-dac/plots/band_indices_distributions_{band}.pdf", dpi=150)

def plot_band_latents(bands, projection='tsne'):
    
    mpl.rc('font',**{'size':11.8})

    # Predefine pastel colors (you can add more if needed)
    pastel_colors = {
        "core": "#FD932C",
        "mb": "#7030A0",
        "hb": "#529D49"
    }
    band_names = {'core': 'Core', 'mb': 'MB', 'hb': 'HB'}
    priority = {"core": 3, "mb": 2, "hb": 1}
    plt.figure(figsize=(9.0, 4.5))

    for band_name, features in bands.items():
        features = features.detach().cpu().squeeze().numpy()
        flat_vals = features.flatten()
        print(band_name, features.shape)
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

    # plt.xlim([-27, 27])
    plt.legend(title="Band Entropy", loc="upper right",
        # bbox_to_anchor=(0.5, -0.1),
        # ncol=1,
        # frameon=False
    )
    # plt.xlabel("Feature Value")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig("/N/slate/daripete/jstsp-dac/plots/band_latents_histogram_combined.pdf", dpi=150)