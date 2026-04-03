
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
import matplotlib.ticker as tck


DIR = '../results/20251023/'
TIME = '20251023_075320'

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


def plot_running_loss(dir=DIR, time=TIME):
    # dir = '../results/20251004/'
    # time = '20251004_072338'
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


def plot_flexcvae_loss(dir=DIR, time=TIME):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    df_train = pd.read_csv(dir + '/losses-flexcvae-train-' + time + '.csv', delimiter=',', header=0)
    df_val = pd.read_csv(dir + '/losses-flexcvae-val-' + time + '.csv', delimiter=',', header=0)
    for col in df_train.columns:
        ax.plot(range(len(df_train[col])), df_train[col], label=col)
    for col in df_val.columns:
        ax.plot(range(len(df_val[col])), df_val[col], label=col)
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
    figname = dir + 'flexcvae_loss_plot_' + time + '.png'
    plt.savefig(figname, dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig(figname.replace('plot_', 'plot_white_'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


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
        ax[i].set_ylabel('Count', fontsize=fontsize)
        ax[i].text(0.95, 0.95, f'{titles[i]}', fontweight='bold',
                   transform=ax[i].transAxes, fontsize=fontsize, va='top', ha='right')
        ax[i].text(0.95, 0.85, f'Mode: {dfmm[t].mode()[0]:.2e}\nMean: {dfmm[t].mean():.2e}\nMedian: {dfmm[t].median():.2e}', 
                   transform=ax[i].transAxes, fontsize=labelsize, va='top', ha='right')
        ax[i].tick_params(which="both", direction='in', top=True, right=True)
        ax[i].tick_params(labelsize=labelsize)
    plt.tight_layout()
    fname = DIR + f'mismatch_hist' + fname
    fname += '-log' if log else ''
    plt.savefig(fname+'-'+TIME+'.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig(fname+'-white'+'-'+TIME+'.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f'Mismatch histograms saved to {DIR}')


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
        plot_mm_hist(dfmm, fontsize=args.fontsize, labelsize=args.labelsize)
        plot_mm_hist(dfmm, log=True, fontsize=args.fontsize, labelsize=args.labelsize)
        plot_mm_hist(dfmmcut, fname='-cut', fontsize=args.fontsize, labelsize=args.labelsize)
        plot_mm_hist(dfmmcut, log=True, fname='-cut', fontsize=args.fontsize, labelsize=args.labelsize)

    if args.contour:
        plot_mmcontour_in_qchi_space(dfmm, fontsize=args.fontsize, labelsize=args.labelsize)
        plot_mmcontour_in_qchi_space(dfmmcut, fname='-cut', fontsize=args.fontsize, labelsize=args.labelsize)

    if args.mm_in_qchi:
        plot_mm_vs_mass(dfmm, fontsize=args.fontsize, labelsize=args.labelsize)
        plot_mm_vs_mass(dfmmcut, fname='-cut', fontsize=args.fontsize, labelsize=args.labelsize)
        plot_mm_vs_chieff(dfmm, fontsize=args.fontsize, labelsize=args.labelsize)
        plot_mm_vs_chieff(dfmmcut, fname='-cut', fontsize=args.fontsize, labelsize=args.labelsize)

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


def plot_rom_opt_mm_hist(fname=None, dir=DIR, time=TIME, fontsize=15, labelsize=13):
    dir = '../results/20260311/'
    time = datetime.now().strftime('%Y%m%d_%H%M%S')
    if fname is None:
        fname = dir + 'mismatch_comparison_results-20260311_002408.csv'
    df = pd.read_csv(fname, skiprows=1,
                     names=['m1', 'm2', 's1', 's2', 'delta_t', 'f_lower', 'mm_rom_hp', 'mm_rom_hc', 'mm_opt_hp', 'mm_opt_hc'])
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    bins = np.logspace(np.log10(df['mm_rom_hp'].min()+1e-10), np.log10(df['mm_rom_hp'].max()+1e-10), 50)
    df.hist('mm_rom_hp', bins=bins, ax=axes[0], grid=False, edgecolor='black', color='salmon')
    df.hist('mm_opt_hp', bins=bins, ax=axes[0], grid=False, edgecolor='black', color='lightgreen', alpha=0.60)
    df.hist('mm_rom_hc', bins=bins, ax=axes[1], grid=False, edgecolor='black', color='salmon')
    df.hist('mm_opt_hc', bins=bins, ax=axes[1], grid=False, edgecolor='black', color='lightgreen', alpha=0.60)
    titles = [ '$\\mathbf{h_{+}}$', '$\\mathbf{h_{\\times}}$']
    for ax, title in zip(axes, titles):
        ax.set_xscale('log')
        ax.tick_params(which="both", direction='in', top=True, right=True)
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.tick_params(labelsize=labelsize)
        ax.set_xscale('log')
        ax.set_xlabel('Mismatch', fontsize=fontsize)
        ax.set_ylabel('Count', fontsize=fontsize)
        ax.legend(['ROM', 'Opt'], fontsize=labelsize)
        ax.text(0.95, 0.95, title, fontweight='bold',
                   transform=ax.transAxes, fontsize=fontsize, va='top', ha='right')
        ax.set_title(None)
    plt.tight_layout()
    savename = dir + 'rom_opt_mm_hist'
    plt.savefig(savename+'-'+time+'.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig(savename+'-white'+'-'+time+'.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    logging.info(f"ROM and Optimized mismatch histograms saved to {savename}")


def plot_uq_hist_from_file(fontsize=15, labelsize=13):
    """
    Plots the hplus and hcross mismatch uncertainty histograms
    on two subplots with log-scaled x-axes. The function reads 
    mismatch data from a specified CSV file, and creates histograms 
    for the hplus and hcross mismatches.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    dir = '../results/20260401/'
    fname = 'uq-test-hist-mean-abs-diff-5000-20260401_174233'
    dfuq = pd.read_csv(dir+fname+'.csv', header=0)
    hpbins = np.logspace(np.log10(dfuq['mmuq_hplus'].min())+1e-10,
                         np.log10(dfuq['mmuq_hplus'].max())+1e-10, 50)
    dfuq.hist('mmuq_hplus', bins=hpbins, ax=axes[0], grid=False, edgecolor='black', color='lightblue')
    dfuq.hist('mmuq_hcross', bins=hpbins, ax=axes[1], grid=False, edgecolor='black', color='lightblue')
    types = ['mmuq_hplus', 'mmuq_hcross']
    titles = [ '$\\mathbf{h_{+}}$', '$\\mathbf{h_{\\times}}$']
    for i in range(len(types)):
        ax = axes[i]
        ax.set_xlim(1e-2,1e0)
        ax.set_xscale('log')
        ax.tick_params(which="both", direction='in', top=True, right=True)
        ax.tick_params(labelsize=labelsize)
        ax.set_xlabel('Mismatch Uncertainty', fontsize=fontsize)
        ax.set_ylabel('Count', fontsize=fontsize)
        ax.text(0.95, 0.95, titles[i], fontweight='bold',
                transform=ax.transAxes, fontsize=labelsize, va='top', ha='right')
        ax.text(0.95, 0.85, f'Mode: {dfuq[types[i]].mode()[0]:.2e}\nMean: {dfuq[types[i]].mean():.2e}\nMedian: {dfuq[types[i]].median():.2e}',
                transform=ax.transAxes, fontsize=labelsize, va='top', ha='right')
        ax.set_title(None)
        ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
    plt.tight_layout()
    savename = dir + 'uq-hphc-hist-mean-abs-diff-5000'
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(savename+'-'+now+'.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig(savename+'-white'+'-'+now+'.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    logging.info(f"Uncertainty histograms saved to {dir}")


def plot_timecompare_from_file(fname=None, dir=DIR, time=TIME, fontsize=12, labelsize=10):
    """
    Plot the time taken to generate different variants of the SEOBNRv4 waveform
    versus the base waveform implementation.
    """
    approximant = 'SEOBNRv4'
    dir = '../results/'
    modeldf = pd.read_csv(dir + '20260228/timecomplexity_results-20260228_031118.csv', skiprows=1,
                          names=['Nruns', 'modeltimes'])
    otherdf = pd.read_csv(dir + '20260126/timecomplexity_compare_20260126_031310.csv', skiprows=1,
                          names=['Nruns', 'basetimes', 'romtimes', 'opttimes'])

    # Plot time taken comparison between model, base, and ROM
    logging.info("Plotting time complexity comparison between model, base, and ROM.")
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(otherdf['Nruns'], otherdf['basetimes'], '.', color='black', markersize=10,
            markeredgewidth=0.5, markeredgecolor='black')
    ax.plot(modeldf['Nruns'], modeldf['modeltimes'], '*', color='blue', markersize=6,
            markeredgewidth=0.15, markeredgecolor='black')
    ax.plot(otherdf['Nruns'], otherdf['romtimes'], '^', color='red', markersize=6,
            markeredgewidth=0.5, markeredgecolor='black')
    ax.plot(otherdf['Nruns'], otherdf['opttimes'], 'v', color='green', markersize=6,
            markeredgewidth=0.5, markeredgecolor='black')
    ax.legend(['base', 'ml', 'ROM', 'opt'], loc='upper right', fontsize=labelsize, title=approximant, title_fontsize=fontsize)
    ax.set_xlabel('Number of Waveforms Generated', fontsize=fontsize)
    ax.set_ylabel('Time (seconds)', fontsize=fontsize)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.xaxis.set_minor_locator(tck.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
    ax.yaxis.set_minor_locator(tck.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
    ax.tick_params(which='both', direction='in', top=True, right=True)
    plt.tight_layout()
    figname = dir + '20260228/' + 'timecomplexity-compare-long-gpu-' + datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(figname+'.png', dpi=300, transparent=True)
    plt.savefig(figname+'-white.png', dpi=300)
    plt.show()
    plt.close()
    logging.info("Time complexity comparison plot saved.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training and validation loss from a file.")
    parser.add_argument('--histogram', action='store_true',
                        help='Plot histograms of mismatch values instead of contour plots.')
    parser.add_argument('--contour', action='store_true',
                        help='Plot contour plots of mismatch values in the mass ratio and chi_eff plane.')
    parser.add_argument('--mm_in_qchi', action='store_true',
                        help='Plot mismatch values against mass parameters and chi_eff.')

    parser.add_argument('--dir', type=str, default='../results/',
                        help='Directory containing the results files. (default:{%(default)s})')
    parser.add_argument('--time', type=str, default=datetime.now().strftime('%Y%m%d-%H%M%S'),
                        help='Timestamp to identify the specific results files to use. (default:{%(default)s})')
    parser.add_argument('--fontsize', type=int, default=15,
                        help='Font size for plot labels and titles. (default:{%(default)s})')
    parser.add_argument('--labelsize', type=int, default=13,
                        help='Font size for plot labels. (default:{%(default)s})')

    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging for debugging purposes.')
    args = parser.parse_args()

    # # save log to a file
    # log_filename = args.dir + 'calc-mismatches-' + args.time + '.log'
    # file_handler = logging.FileHandler(log_filename)
    # file_handler.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # file_handler.setFormatter(formatter)
    # logging.getLogger().addHandler(file_handler)


    logging.basicConfig(level=logging.INFO if args.verbose == 0 else logging.DEBUG, 
                        format='%(asctime)s - %(levelname)s - %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S')


    # plot_running_loss(dir=args.dir, time=args.time)
    mismatch_anal(args)
    # plot_rom_opt_mm_hist(fontsize=20, labelsize=15)
    # plot_timecompare_from_file()
    # plot_flexcvae_loss(dir=args.dir, time=args.time)
    # plot_uq_hist_from_file(fontsize=20, labelsize=15)
