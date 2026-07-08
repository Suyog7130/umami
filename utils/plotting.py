

import numpy as np
from datetime import datetime

# import matplotlib
# matplotlib.use('Agg')   # non GUI backend
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

import logging
logger = logging.getLogger(__name__)


def _plot_zoom_window(ax, x, y, y2=None, zoom_halfwidth=75, padding=20):
    zoom_start = max(0, np.argmax(y) - zoom_halfwidth)
    zoom_end = min(len(y)-1, np.argmax(y) + zoom_halfwidth)
    ax.plot(x[zoom_start:zoom_end], y[zoom_start:zoom_end], color='blue')
    if y2 is not None:
        ax.plot(x[zoom_start:zoom_end], y2[zoom_start:zoom_end], color='orange')
    yrange = max(y[zoom_start:zoom_end]) - min(y[zoom_start:zoom_end])
    ypad = yrange * padding / 100  # Convert percentage to actual padding
    ax.set_xlim(x[zoom_start], x[zoom_end])
    ax.set_ylim(min(y[zoom_start:zoom_end]) - ypad, max(y[zoom_start:zoom_end]) + ypad)


def plot_twopanel(xarr: np.ndarray, 
                yarr: list[dict[str, np.ndarray], dict[str, np.ndarray]],
                title=None, 
                axes_labels=['Time (s)', 'Amplitude', 'Frequency (Hz)'],
                with_zoom_windows=True,
                savename=None, **kwargs):
    """
    Plot two panels plot, with 2 more smaller zoomed windows!

    Arguments
    ---------
    xarr: np.ndarray
        The x-axis array, which is the same for all the curves being plotted.
    yarr: list of dicts
        A list of two dictionaries, where each dictionary contains the y-axis arrays for the curves to be plotted in each panel. The keys of the dictionaries are the labels for the curves, and the values are the corresponding y-axis arrays. The first dictionary corresponds to the first panel, and the second dictionary corresponds to the second panel.
    """
    fontsize = kwargs.get('fontsize', 12)
    labelsize = kwargs.get('labelsize', 10)

    if with_zoom_windows:
        fig, axes = plt.subplots(2, 2, figsize=(14, 6), width_ratios=[3, 1])
        axes = axes.flatten()    
        for ax in [axes[0], axes[2]]:
            ax.sharex(axes[0])
        fullsize_axes = [axes[0], axes[2]]
        # Zoomed view near the maximum
        _plot_zoom_window(axes[1], xarr, list(yarr[0].values())[0], y2=list(yarr[0].values())[1])
        _plot_zoom_window(axes[3], xarr, list(yarr[1].values())[0], y2=list(yarr[1].values())[1])
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fullsize_axes = axes

    for i, ax in enumerate(fullsize_axes):
        ax.plot(xarr, list(yarr[i].values())[0], label=list(yarr[i].keys())[0], color='blue')
        if len(yarr[i]) > 1:
            ax.plot(xarr, list(yarr[i].values())[1], label=list(yarr[i].keys())[1], color='orange')

    for i in range(len(axes)):
        axes[i].set_xlabel(axes_labels[0], fontsize=fontsize)
        axes[i].xaxis.set_minor_locator(tck.AutoMinorLocator())
        axes[i].yaxis.set_minor_locator(tck.AutoMinorLocator())
        axes[i].tick_params(which='both', direction='in', top=True, right=True)
        axes[i].tick_params(axis='both', labelsize=labelsize)
    fullsize_axes[0].set_ylabel(axes_labels[1], fontsize=fontsize)
    fullsize_axes[1].set_ylabel(axes_labels[2], fontsize=fontsize)
    fullsize_axes[0].legend(fontsize=labelsize, loc='upper left')
    fullsize_axes[1].legend(fontsize=labelsize, loc='upper left')

    if kwargs.get('logscale', False):
        for ax in fullsize_axes:
            ax.set_yscale('symlog', linthresh=1e-6)  # Set y-axis to symmetric log scale with a linear threshold

    # Place suptitle closer to the top of the figure, not too far away
    plt.suptitle(title, fontsize=fontsize,
        y=0.97  # Move title closer to the top edge (default is 0.99)
    )

    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.2)
    # putils.beautifyPlot(axes)
    if savename is not None:
        if not savename.endswith('.png'):
            savename += '.png'
        if kwargs.get('transparent', False):
            plt.savefig(savename, dpi=300, bbox_inches='tight', transparent=True)
        else:
            plt.savefig(savename, dpi=300, bbox_inches='tight')
        logger.info(f"Overplot saved to {savename}")
    if kwargs.get('showplot', False):
        plt.show()
    plt.close('all')
