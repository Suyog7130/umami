
import argparse
import matplotlib.pyplot as plt
import numpy as np
from plotutils import putils

def plot(ax, fname, logscale=False, save=False, show=True,
         label=''):
    data = np.loadtxt(fname, delimiter=',', skiprows=1)
    ax.plot(range(len(data)), data)
    if logscale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    if save:
        plt.savefig(fname + '-plot.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training and validation loss from a file.")
    parser.add_argument('fname', type=str, default='loss',
                        help='Path to the file containing loss data.')
    
    dir = 'results/'
    time = '20250523_095852'
    tloss = dir + 'train_loss_' + time + '.txt'
    trloss = dir + 'train_rloss_' + time + '.txt'
    vloss = dir + 'valid_loss_' + time + '.txt'
    vrloss = dir + 'valid_rloss_' + time + '.txt'

    fig, axes = plt.subplots(1, 2, figsize=(10,5))

    plot(axes[0], tloss, logscale=False, show=False)
    plot(axes[0], vloss, logscale=False, show=False)
    axes[0].set_title('Epoch Loss', fontsize=12)
    axes[0].set_xlabel('Epoch', fontsize=12)
    plot(axes[1], trloss, logscale=True, show=False)
    plot(axes[1], vrloss, logscale=True, show=False)
    axes[1].set_title('Running Loss', fontsize=12)
    axes[1].set_xlabel('Batch', fontsize=12)
    for i in range(len(axes)):
        axes[i].set_ylabel('Loss', fontsize=12)
        axes[i].legend(['Train Loss', 'Validation Loss'])

    plt.tight_layout()
    putils.beautifyPlot([axes[0]])
    figname = dir + 'loss_plot_' + time + '.png'
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    plt.show()
