import numpy as np
import matplotlib.pyplot as plt

def plot_affinity(lookup,molecule_list):

    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(lookup, vmin=-1, vmax=1, cmap=plt.cm.get_cmap('RdBu'))
    ax.set_title('Shape similarity matrix')
    ax.set_xticks(np.arange(0, len(molecule_list)))
    ax.set_xticklabels(molecule_list)
    ax.set_yticks(np.arange(0, len(molecule_list)))
    ax.set_yticklabels(molecule_list)
    ax.tick_params(axis='x', rotation=90)
    fig.colorbar(im, ax=ax)
    plt.savefig('similarity.png', dpi=144)