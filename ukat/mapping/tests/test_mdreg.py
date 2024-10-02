import os
import numpy as np
import matplotlib.pyplot as plt
import mdreg

from ukat.mapping.t1 import T1


def t1_maps():

    # fetch data
    data = mdreg.fetch('MOLLI')
    array = data['array'][:, :, 0, :]
    TI = np.array(data['TI'])

    # Calculate corrected uncorrected T1-map
    map = T1(array, TI, np.eye(4))
    np.save(os.path.join(os.getcwd(), 't1_uncorr'), map.t1_map)

    # Calculate corrected corrected T1-map
    map = T1(array, TI, np.eye(4), mdr=True)
    np.save(os.path.join(os.getcwd(), 't1_corr'), map.t1_map)


def plot_t1_maps():

    # Plot corrected and uncorrected side by side
    t1_uncorr = np.load(os.path.join(os.getcwd(), 't1_uncorr.npy'))
    t1_corr = np.load(os.path.join(os.getcwd(), 't1_corr.npy'))
    fig, ax = plt.subplots(figsize=(10, 6), ncols=2, nrows=1)
    for i in range(2):
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])
    ax[0].set_title('T1-map without MDR')
    ax[1].set_title('T1-map with MDR')
    im = ax[0].imshow(t1_uncorr.T, cmap='gray', vmin=0, vmax=2000)
    im = ax[1].imshow(t1_corr.T, cmap='gray', vmin=0, vmax=2000)
    fig.colorbar(im, ax=ax.ravel().tolist())
    plt.savefig(os.path.join(os.getcwd(), 't1_mdr'))
    plt.show()


if __name__ == '__main__':

    # t1_maps()
    plot_t1_maps()
