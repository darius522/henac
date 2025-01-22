import os

from matplotlib import pyplot as plt
from scipy.stats import entropy
import numpy as np

import warnings

def residual_plots(x, dir):
    warnings.warn(
        f"Feature statistics are currently being computed in encoder."
    )
    _, axes = plt.subplots(ncols=1, nrows=1, figsize=(8,4))
    xh = np.histogram(x.flatten().detach().cpu().numpy(), bins=64, range=(-6,6), density=True)[0]
    full_entropy = entropy(xh).item()
    axes.hist(x.detach().cpu().numpy().flatten(),facecolor='r', alpha=.5, linewidth=0, label=f'no diff, entropy={"{:0.2f}".format(full_entropy)}', bins=64, range=[-6,6])
    axes.legend()
    from random import randint
    rand = randint(0,1000)
    plt.tight_layout()
    plt.savefig(os.path.join(dir, f'{rand}.png'))