import os

from matplotlib import pyplot as plt
from scipy.stats import entropy
import numpy as np

import warnings

def residual_plots(x, r, i):
    warnings.warn(
        f"Feature statistics are currently being computed in encoder."
    )
    _, axes = plt.subplots(ncols=1, nrows=1, figsize=(8,4))
    xh = np.histogram(x.flatten().detach().cpu().numpy(), bins=64, range=(-6,6), density=True)[0]
    rh = np.histogram(r.flatten().detach().cpu().numpy(), bins=64, range=(-6,6), density=True)[0]
    full_entropy = entropy(xh).item()
    resi_entropy = entropy(rh).item()
    axes.hist(x.detach().cpu().numpy().flatten(),facecolor='r', alpha=.5, linewidth=0, label=f'full, entropy={"{:0.2f}".format(full_entropy)}', bins=64, range=[-6,6])
    axes.hist(r.detach().cpu().numpy().flatten(),facecolor='g', alpha=.5, linewidth=0, label=f'residual, entropy={"{:0.2f}".format(resi_entropy)}', bins=64, range=[-6,6])
    axes.legend()
    from random import randint
    rand = randint(0,1000)
    plt.tight_layout()
    plt.savefig(os.path.join('/N/slate/daripete/jstsp-dac/plots', f'{rand}_{i}_{list(x.shape)}.png'))