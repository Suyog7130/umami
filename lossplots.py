
import os
import h5py
import logging
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.interpolate import griddata
from matplotlib.colors import LogNorm
from datetime import datetime


DIR = '../results/20251023/'
TIME = '20251023_075320'


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S')

# save log to a file
log_filename = DIR + 'calc-mismatches-' + TIME + '.log'
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logging.getLogger().addHandler(file_handler)


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


def plot_running_loss():
    dir = '../results/20251004/'
    time = '20251004_072338'
    trloss = np.loadtxt(dir + 'train-rloss-' + time + '.txt', delimiter=',', skiprows=1)
    vrloss = np.loadtxt(dir + 'valid-rloss-' + time + '.txt', delimiter=',', skiprows=1)
    netloss = np.loadtxt(dir + 'net-loss-' + time + '.csv', delimiter=',', skiprows=1)
    reconloss = netloss[:,0]
    klloss = netloss[:,1]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(range(len(trloss)), trloss, label='Train Running Loss')
    ax.plot(range(len(vrloss)), vrloss, label='Valid Running Loss')
    ax.plot(range(len(reconloss)), reconloss, label='Reconstruction Loss')
    ax.plot(range(len(klloss)), klloss, label='KL Divergence Loss')
    ax.set_xlabel('Cumulative Steps', fontsize=12)
    ax.set_ylabel('Running Loss', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1, 10**4)
    ax.xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
    ax.yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.legend()
    plt.tight_layout()
    figname = dir + 'running_loss_plot_' + time + '.png'
    plt.savefig(figname, dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig(figname.replace('plot_', 'plot_white_'), dpi=300, bbox_inches='tight')
    plt.close()


  
    # dir = '../results/20250813/'
    # time = '20250813_033457-10'
    # tloss = dir + 'train_loss_' + time + '.txt'
    # trloss = dir + 'train_rloss_' + time + '.txt'
    # vloss = dir + 'valid_loss_' + time + '.txt'
    # vrloss = dir + 'valid_rloss_' + time + '.txt'

    # fig, axes = plt.subplots(1, 2, figsize=(10,5))

    # plot(axes[0], tloss, logscale=False, show=False)
    # plot(axes[0], vloss, logscale=False, show=False)
    # axes[0].set_title('Epoch Loss', fontsize=12)
    # axes[0].set_xlabel('Epoch', fontsize=12)
    # plot(axes[1], trloss, logscale=True, show=False)
    # plot(axes[1], vrloss, logscale=True, show=False)
    # axes[1].set_title('Running Loss', fontsize=12)
    # axes[1].set_xlabel('Batch', fontsize=12)
    # for i in range(len(axes)):
    #     axes[i].set_ylabel('Loss', fontsize=12)
    #     axes[i].legend(['Train Loss', 'Validation Loss'])
    #     axes[i].xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
    #     axes[i].yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
    # plt.tight_layout()
    # putils.beautifyPlot([axes[0]])
    # figname = dir + 'loss_plot_' + time + '.png'
    # plt.savefig(figname, dpi=300, bbox_inches='tight')
    # plt.show()


def plot_mm_hist(dfmm, log=False, fontsize=15, labelsize=13, fname=''):
    """
    Plot histograms of mismatch values for different types of mismatches. 
    The function takes a DataFrame containing mismatch data and creates histograms 
    for amplitude, frequency, h_plus, and h_cross mismatches. The x-axis is set to 
    logarithmic scale, and the y-axis can also be set to logarithmic scale based on 
    the 'log' parameter. Each subplot includes the mode, mean, and median of the 
    mismatch values for that type.
    """
    types = ['mismatch_amp', 'mismatch_freq', 'mismatch_hplus', 'mismatch_hcross']
    titles = ['Amplitude', 'Frequency', '$\\mathbf{h_{+}}$', '$\\mathbf{h_{\\times}}$']
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.flatten()
    for i, t in enumerate(types):
        bins = np.logspace(np.log10(dfmm[t].min()+1e-10), np.log10(dfmm[t].max()+1e-10), 50)
        dfmm.hist(t, bins=bins, ax=ax[i], grid=False, edgecolor='black', color='skyblue')
        ax[i].set_title(None)
        ax[i].set_xscale('log')
        if log:
            ax[i].set_yscale('log')
        ax[i].set_xlabel('Mismatch', fontsize=fontsize)
        ax[i].set_ylabel('Frequency', fontsize=fontsize)
        ax[i].text(0.95, 0.95, f'{titles[i]}', fontweight='bold',
                   transform=ax[i].transAxes, fontsize=fontsize, va='top', ha='right')
        ax[i].text(0.95, 0.85, f'Mode: {dfmm[t].mode()[0]:.2e}\nMean: {dfmm[t].mean():.2e}\nMedian: {dfmm[t].median():.2e}', 
                   transform=ax[i].transAxes, fontsize=fontsize, va='top', ha='right')
        ax[i].tick_params(which="both", direction='in', top=True, right=True)
        ax[i].tick_params(labelsize=labelsize)
    plt.tight_layout()
    fname = DIR + f'mismatch_hist' + fname
    fname += '-log' if log else ''
    plt.savefig(fname+'-'+TIME+'.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig(fname+'-white'+'-'+TIME+'.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f'Mismatch histograms saved to {DIR}')


def plot_mmcontour_in_qchi_space(dfmm, fontsize=15, labelsize=13, fname=''):
        """
        Plot the mismatches in the mass ratio and chi_eff plane as contours.
        """
        logging.info("Plotting mismatches in the mass ratio and chi_eff plane.")
        titles = ['Amplitude', 'Frequency', '$\\mathbf{h_{+}}$', '$\\mathbf{h_{\\times}}$']

        # Create a grid of points
        q = dfmm['mass_ratio'].values
        chi = dfmm['chi_eff'].values
        xi = np.linspace(min(q), max(q), 100)
        yi = np.linspace(min(chi), max(chi), 100)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate mismatch values onto the grid
        zi_amp = griddata((q, chi), dfmm['mismatch_amp'].values, (xi, yi), method='linear')
        zi_freq = griddata((q, chi), dfmm['mismatch_freq'].values, (xi, yi), method='linear')
        zi_hplus = griddata((q, chi), dfmm['mismatch_hplus'].values, (xi, yi), method='linear')
        zi_hcross = griddata((q, chi), dfmm['mismatch_hcross'].values, (xi, yi), method='linear')
        data = [zi_amp, zi_freq, zi_hplus, zi_hcross]

        # Plotting
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        contour_levels = np.logspace(-6, 0, 13)

        for i, ax in enumerate(axes.flat):
            cs = ax.contourf(xi, yi, data[i], levels=contour_levels, norm=LogNorm(), cmap='viridis')
            fig.colorbar(cs, ax=ax, label=f'{titles[i]} mismatch')
            # ax.set_title(titles[i], fontsize=fontsize, fontweight='bold')
            ax.set_xlabel('q', fontsize=fontsize)
            ax.set_ylabel('$\\chi_{\\rm eff}$', fontsize=fontsize)
            ax.minorticks_on()
            ax.tick_params(which='both', direction='in', top=True, right=True)
            ax.tick_params(labelsize=labelsize)

        plt.tight_layout()
        savename = DIR + f'mmcontour_in_qchi'
        savename += fname if fname else ''
        plt.savefig(savename+'-'+TIME+'.png', dpi=300, bbox_inches='tight', transparent=True)
        plt.savefig(savename+'-white'+'-'+TIME+'.png', dpi=300, bbox_inches='tight')
        logging.info(f"Mismatch in q-chi_eff plane plot saved to {savename}")
        plt.close()


def plot_mm_vs_mass(dfmm, fontsize=15, labelsize=13, fname=''):
    kinds = ['chirp_mass', 'total_mass', 'mass_ratio']
    types = [['mismatch_amp', 'mismatch_freq'],
             ['mismatch_hplus', 'mismatch_hcross']]
    titles = [['Amplitude', 'Frequency'], 
              ['$\\mathbf{h_{+}}$', '$\\mathbf{h_{\\times}}$']]

    for kind in kinds:
        fig, ax = plt.subplots(1,2,figsize=(10,5))
        for i in range(len(types)):
            ax[i].plot(dfmm[kind], dfmm[types[i][0]], 'o', label=titles[i][0],
                        markersize=3, alpha=0.5, markeredgewidth=0.25, markeredgecolor='black')
            ax[i].plot(dfmm[kind], dfmm[types[i][1]], 's', label=titles[i][1],
                        markersize=3, alpha=0.5, markeredgewidth=0.25, markeredgecolor='black')
            ax[i].set_xlabel(kind.capitalize().replace('_', ' '), fontsize=fontsize)
            ax[i].set_ylabel('Mismatch', fontsize=fontsize)
            ax[i].set_yscale('log')
            ax[i].minorticks_on()
            ax[i].tick_params(which='both', direction='in', top=True, right=True)
            ax[i].tick_params(labelsize=labelsize)
            ax[i].legend(fontsize=labelsize)
        savename = DIR + f'mm_vs_{kind}'
        savename += fname if fname else ''
        plt.tight_layout()
        plt.savefig(savename+'-'+TIME+'.png', dpi=300, bbox_inches='tight', transparent=True)
        plt.savefig(savename+'-white'+'-'+TIME+'.png', dpi=300, bbox_inches='tight')
        logging.info(f"Mismatch vs {kind} plot saved to {savename}")
        plt.close()


def plot_mm_vs_chieff(dfmm, fontsize=15, labelsize=13, fname=''):
    types = [['mismatch_amp', 'mismatch_freq'],
             ['mismatch_hplus', 'mismatch_hcross']]
    titles = [['Amplitude', 'Frequency'], 
              ['$\\mathbf{h_{+}}$', '$\\mathbf{h_{\\times}}$']]
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for i in range(len(types)):
        ax[i].plot(dfmm['chi_eff'], dfmm[types[i][0]], 'o', label=titles[i][0],
                   markersize=3, alpha=0.5, markeredgewidth=0.25, markeredgecolor='black')
        ax[i].plot(dfmm['chi_eff'], dfmm[types[i][1]], 's', label=titles[i][1],
                   markersize=3, alpha=0.5, markeredgewidth=0.25, markeredgecolor='black')
        ax[i].set_xlabel('$\\chi_{\\rm eff}$', fontsize=fontsize)
        ax[i].set_ylabel('Mismatch', fontsize=fontsize)
        ax[i].set_yscale('log')
        ax[i].minorticks_on()
        ax[i].tick_params(which='both', direction='in', top=True, right=True)
        ax[i].tick_params(labelsize=labelsize)
        ax[i].legend(fontsize=labelsize)
    savename = DIR + f'mm_vs_chieff'
    savename += fname if fname else ''
    plt.tight_layout()
    plt.savefig(savename+'-'+TIME+'.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig(savename+'-white'+'-'+TIME+'.png', dpi=300, bbox_inches='tight')
    logging.info(f"Mismatch vs chi_eff plot saved to {savename}")
    plt.close()


def apply_chi_cuts(dfmm, cut=0.8):
    """
    Apply cuts to the DataFrame based on chi_eff values.
    """
    chi_range = [-cut, cut]
    dfmmcut = dfmm[(dfmm['chi_eff'] >= chi_range[0]) & (dfmm['chi_eff'] <= chi_range[1])]
    logging.info(f'Selected {len(dfmmcut)} samples within chi_eff range {chi_range}')
    return dfmmcut


def mismatch_anal(args, cut=0.8):
    logging.info("Starting mismatch analysis...")
    mmfile = DIR + 'mismatch-results-' + TIME + '.h5'
    dfmm = pd.read_hdf(mmfile, key='dfmm')
    # print(dfmm.head())
    logging.info(f"Columns in mismatch DataFrame: {dfmm.columns.tolist()}")
    # print(dfmm.describe())

    # Select regions of interest in parameter space
    chi_range = [-cut, cut]
    dfmmcut = dfmm[(dfmm['chi_eff'] >= chi_range[0]) & (dfmm['chi_eff'] <= chi_range[1])]
    logging.info(f'Selected {len(dfmmcut)} samples within chi_eff range {chi_range}')
    logging.info(f'Original dataset size: {len(dfmm)} samples')

    # Plot mismatch histograms
    if args.histogram:
        plot_mm_hist(dfmm)
        plot_mm_hist(dfmm, log=True)
        plot_mm_hist(dfmmcut, fname='-cut')
        plot_mm_hist(dfmmcut, log=True, fname='-cut')

    if args.contour:
        plot_mmcontour_in_qchi_space(dfmm)
        plot_mmcontour_in_qchi_space(dfmmcut, fname='-cut')

    if args.mm_in_qchi:
        plot_mm_vs_mass(dfmm)
        plot_mm_vs_mass(dfmmcut, fname='-cut')
        plot_mm_vs_chieff(dfmm)
        plot_mm_vs_chieff(dfmmcut, fname='-cut')

    # Output mean, median, best & worst mismatches
    logging.info("Calculating mean, median, best & worst mismatches...")
    for t in ['mismatch_amp', 'mismatch_freq', 'mismatch_hplus', 'mismatch_hcross']:
        mean_mm = dfmm[t].mean()
        median_mm = dfmm[t].median()
        best_mm = dfmm[t].min()
        worst_mm = dfmm[t].max()
        logging.info(f"{t}: Mean = {mean_mm:.2e}, Median = {median_mm:.2e}, Best = {best_mm:.2e}, Worst = {worst_mm:.2e}")
        mean_mm_cut = dfmmcut[t].mean()
        median_mm_cut = dfmmcut[t].median()
        best_mm_cut = dfmmcut[t].min()
        worst_mm_cut = dfmmcut[t].max()
        logging.info(f"{t} (cut): Mean = {mean_mm_cut:.2e}, Median = {median_mm_cut:.2e}, Best = {best_mm_cut:.2e}, Worst = {worst_mm_cut:.2e}")
    logging.info("Mismatch analysis completed.")

    # Now, find best & worst mismatches at different chi_eff cuts
    logging.info("Analyzing mismatches at different chi_eff cuts...")
    chi_cuts = [0.5, 0.6, 0.75, 0.8]
    for cut in chi_cuts:
        dfmmcut = apply_chi_cuts(dfmm, cut=cut)
        logging.info(f"Chi_eff cut: ±{cut}")
        for t in ['mismatch_amp', 'mismatch_freq', 'mismatch_hplus', 'mismatch_hcross']:
            best_mm_cut = dfmmcut[t].min()
            worst_mm_cut = dfmmcut[t].max()
            logging.info(f"{t} (cut ±{cut}): Best = {best_mm_cut:.2e}, Worst = {worst_mm_cut:.2e}")
    logging.info("Mismatch analysis completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training and validation loss from a file.")
    parser.add_argument('--histogram', action='store_true',
                        help='Plot histograms of mismatch values instead of contour plots.')
    parser.add_argument('--contour', action='store_true',
                        help='Plot contour plots of mismatch values in the mass ratio and chi_eff plane.')
    parser.add_argument('--mm_in_qchi', action='store_true',
                        help='Plot mismatch values against mass parameters and chi_eff.')
    args = parser.parse_args()

    # plot_running_loss()
    mismatch_anal(args)
