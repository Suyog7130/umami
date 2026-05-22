
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


def plot_loss_from_file(fontsize=12, val_plot_type: {'scatter', None} = None,
                        onlyprintsteps=False):
    dir = '../results/20260408/'
    time = '20260408_062713'
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # Plot running loss from file
    trloss = np.loadtxt(dir + 'train-rloss-' + time + '.txt', delimiter=',', skiprows=1)
    traindf = pd.read_csv(dir + 'net-train-loss-' + time + '.csv', delimiter=',', header=0)
    evaldf = pd.read_csv(dir + 'train-eval-loss-' + time + '.csv', delimiter=',', header=0)
    # Plot validation losses
    vrloss = np.loadtxt(dir + 'valid-rloss-' + time + '.txt', delimiter=',', skiprows=1)
    validdf = pd.read_csv(dir + 'net-val-loss-' + time + '.csv', delimiter=',', header=0)

    # Plot and save figs in the directory
    ax.plot(range(len(trloss)), trloss, label='Total Train Loss')
    ax.plot(range(len(vrloss)), vrloss, label='Total Valid Loss')
    for col in traindf.columns:
        ax.plot(range(len(traindf[col])), traindf[col], label=f'Train {col}')
    for col in evaldf.columns:
        ax.plot(range(len(evaldf[col])), evaldf[col], label=f'Eval {col}')
    for col in validdf.columns:
        ax.plot(range(len(validdf[col])), validdf[col], label=f'Valid {col}')
    ax.set_xlabel('Cumulative Steps', fontsize=fontsize)
    ax.set_ylabel('Loss', fontsize=fontsize)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
    ax.yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
    ax.tick_params(which='both', direction='in', top=True, right=True)
    plt.tight_layout()
    figname = dir + 'loss_plot_' + time + '.png'
    # plt.savefig(figname, dpi=300, bbox_inches='tight', transparent=True)
    # plt.show()
    plt.close()
    logging.info(f"Loss plot saved to {figname}")

    # -- Now, plot train, eval, valid losses in one plot each for total, recon and kl losses
    neteval = evaldf['train_eval_loss']
    reconloss_train = traindf['netreconloss']
    reconloss_eval = evaldf['train_eval_recon_loss']
    reconloss_valid = validdf['netvreconloss']
    klloss_train = traindf['netklloss']
    klloss_eval = evaldf['train_eval_kl_loss']
    klloss_valid = validdf['netvklloss']

    num_epochs = 10  # -- this is fixed!
    train_steps_per_epoch = len(trloss) // num_epochs
    logging.info(f"Train steps per epoch: {train_steps_per_epoch}")
    eval_steps_per_epoch = len(neteval) // num_epochs
    val_steps_per_epoch = len(vrloss) // num_epochs
    logging.info(f"Eval steps per epoch: {eval_steps_per_epoch}")
    logging.info(f"Valid steps per epoch: {val_steps_per_epoch}")
    if onlyprintsteps:
        print(f"Train steps per epoch: {train_steps_per_epoch}")
        print(f"Eval steps per epoch: {eval_steps_per_epoch}")
        print(f"Valid steps per epoch: {val_steps_per_epoch}")
        return
    
    items = [{'quant':[trloss, vrloss, neteval],
                'label':['Train: Total Loss', 'Valid: Total Loss', 'Eval: Total Loss'],
                'savename':'total_loss_plot_'},
            {'quant':[reconloss_train, reconloss_eval, reconloss_valid],
                'label':['Train: Recon Loss', 'Eval: Recon Loss', 'Valid: Recon Loss'],
                'savename':'recon_loss_plot_'},
            {'quant':[klloss_train, klloss_eval, klloss_valid],
                'label':['Train: KL Loss', 'Eval: KL Loss', 'Valid: KL Loss'],
                'savename':'kl_loss_plot_'},
            {'quant':[klloss_train, reconloss_train, trloss],
                'label':['Train: KL Loss', 'Train: Recon Loss', 'Train: Total Loss'],
                'savename':'train_loss_plot_'},
            {'quant':[vrloss, reconloss_valid, klloss_valid],
                'label':['Valid: Total Loss', 'Valid: Recon Loss', 'Valid: KL Loss'],
                'savename':'valid_loss_plot_'},
            {'quant':[neteval, reconloss_eval, klloss_eval],
                'label':['Eval: Total Loss', 'Eval: Recon Loss', 'Eval: KL Loss'],
                'savename':'eval_loss_plot_'},
            {'quant':[trloss, neteval, reconloss_train, reconloss_eval, klloss_train, klloss_eval],
                'label':['Train: Total Loss', 'Eval: Total Loss', 'Train: Recon Loss', 'Eval: Recon Loss',
                            'Train: KL Loss', 'Eval: KL Loss'],
                'savename':'train_eval_loss_plot_'},
            {'quant':[vrloss, neteval, reconloss_valid, reconloss_eval, klloss_valid, klloss_eval],
                'label':['Valid: Total Loss', 'Eval: Total Loss', 'Valid: Recon Loss', 'Eval: Recon Loss',
                        'Valid: KL Loss', 'Eval: KL Loss'],
                'savename':'valid_eval_loss_plot_'},
            {'quant':[trloss, vrloss, reconloss_train, reconloss_valid, klloss_train, klloss_valid],
                'label':['Train: Total Loss', 'Valid: Total Loss', 'Train: Recon Loss', 'Valid: Recon Loss',
                        'Train: KL Loss', 'Valid: KL Loss'],
                'savename':'train_valid_loss_plot_'},
            {'quant':[trloss, vrloss, neteval, klloss_train, klloss_valid, klloss_eval],
                'label':['Train: Total Loss', 'Valid: Total Loss', 'Eval: Total Loss', 'Train: KL Loss',
                        'Valid: KL Loss', 'Eval: KL Loss'],
                'savename':'total_kl_loss_plot_'},
            {'quant':[trloss, vrloss, neteval, reconloss_train, reconloss_valid, reconloss_eval],
                'label':['Train: Total Loss', 'Valid: Total Loss', 'Eval: Total Loss', 'Train: Recon Loss',
                        'Valid: Recon Loss', 'Eval: Recon Loss'],
                'savename':'total_recon_loss_plot_'}
            ]
    for item in items:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        for i in range(len(item['quant'])):
            # -- If `plot_val_vert` is on, then plot validation loss as vertically distributed points,
            # after each epoch worth of training steps, instead of spreading them horizontally!
            if val_plot_type is not None:
                # -- Iterate over number of epochs, plotting val loss at each epoch end point
                if item['label'][i].startswith('Valid'):
                    steps_per_epoch = val_steps_per_epoch
                    if 'Total' in item['label'][i]:
                        color = 'orange'
                    elif 'Recon' in item['label'][i]:
                        color = 'cyan'
                    elif 'KL' in item['label'][i]:
                        color = 'magenta'
                elif item['label'][i].startswith('Eval'):
                    steps_per_epoch = eval_steps_per_epoch
                    if 'Total' in item['label'][i]:
                        color = 'red'
                    elif 'Recon' in item['label'][i]:
                        color = 'blue'
                    elif 'KL' in item['label'][i]:
                        color = 'purple'
                else:
                    # -- For train losses, just plot as usual
                    ax.plot(range(len(item['quant'][i])), item['quant'][i], label=item['label'][i], alpha=0.6)
                    continue
                logging.info(f"Plotting {item['label'][i]} with {num_epochs} epochs and {steps_per_epoch} steps per epoch.")
                for epoch in range(num_epochs):
                    epoch_end_step = (epoch + 1) * train_steps_per_epoch
                    losses = item['quant'][i][epoch * steps_per_epoch : (epoch + 1) * steps_per_epoch]
                    # -- add label the first time and then set it to None for subsequent epochs to avoid duplicate legend entries
                    if epoch == 0:
                        ax.plot([epoch_end_step] * len(losses), losses, '.', alpha=0.6, color=color, label=item['label'][i])
                    else:
                        ax.plot([epoch_end_step] * len(losses), losses, '.', alpha=0.6, color=color)
            else:
                ax.plot(range(len(item['quant'][i])), item['quant'][i], label=item['label'][i], alpha=0.6)
        ax.set_xlabel('Cumulative Steps', fontsize=fontsize)
        ax.set_ylabel('Loss', fontsize=fontsize)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
        ax.yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
        ax.tick_params(which='both', direction='in', top=True, right=True)
        plt.tight_layout()
        figname = dir + item['savename']
        figname += f'val-{val_plot_type}-' if val_plot_type is not None else ''
        plt.savefig(figname+f'{time}.png', dpi=300, bbox_inches='tight', transparent=True)
        # plt.show()
        plt.close()
        logging.info(f"{item['savename'][:-1]} loss plot saved to {figname}")

    # -- Plot Validation / Eval losses as bars in a separate figure!
    # -- Plot bars of losses for each epoch at appropriate num of steps centered vertically at the mean loss for that epoch, 
    # with error bars showing the std deviation of the losses in that epoch!
    # Each epoch will show all items to be plotted beside each other with some width, 
    # and spacing. Since this is a different figure, the xaxis can be different, for instance having
    # number of epochs on bottom xaxis with each epoch's vertical location been a discrete point, and
    # the cumulative steps been shown on secondary top xaxis in log-scale, simply for reference!
    for item in items:
        if not any(label.startswith('Valid') or label.startswith('Eval') for label in item['label']):
            continue  # -- only plot bars for items that have valid or eval losses
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        for i in range(len(item['quant'])):
            if item['label'][i].startswith('Valid'):
                steps_per_epoch = val_steps_per_epoch
                if 'Total' in item['label'][i]:
                    color = 'orange'
                elif 'Recon' in item['label'][i]:
                    color = 'cyan'
                elif 'KL' in item['label'][i]:
                    color = 'magenta'
            elif item['label'][i].startswith('Eval'):
                steps_per_epoch = eval_steps_per_epoch
                if 'Total' in item['label'][i]:
                    color = 'red'
                elif 'Recon' in item['label'][i]:
                    color = 'blue'
                elif 'KL' in item['label'][i]:
                    color = 'purple'
            else:
                continue  # -- only plot bars for valid and eval losses
            for epoch in range(num_epochs):
                losses = item['quant'][i][epoch * steps_per_epoch : (epoch + 1) * steps_per_epoch]
                mean_loss = np.mean(losses)
                std_loss = np.std(losses)
                bar_width = 0.8 / len(item['quant'])  # -- width of each bar, with some spacing
                bar_x = epoch + 1 + (i - len(item['quant']) / 2) * bar_width + bar_width / 2  # -- center bars around epoch number
                ax.bar(bar_x, height=std_loss*2, bottom=mean_loss-std_loss, width=bar_width, color=color, alpha=0.6, 
                       label=item['label'][i] if epoch == 0 else None)
        ax.set_yscale('log')
        ax.set_xlabel('Epoch', fontsize=fontsize)
        ax.set_ylabel('Loss', fontsize=fontsize)
        ax.legend()
        ax.xaxis.set_major_locator(plt.MaxNLocator(11))  # -- set major ticks at epoch numbers
        ax.yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
        ax.tick_params(which='both', direction='in', top=True, right=True)
        # # -- add num of training steps on secondary x-axis at the top in log scale for reference, 
        # # assuming each epoch has same number of training steps, which is true in our case since 
        # # we are plotting cumulative steps on x-axis in the previous plots!
        # secax = ax.secondary_xaxis('top', functions=(lambda epoch: epoch * train_steps_per_epoch, lambda steps: steps / train_steps_per_epoch))
        # secax.tick_params(which='both', direction='in', top=True)
        # secax.set_xscale('log')
        # secax.set_xlabel('Cumulative Training Steps', fontsize=fontsize)
        figname = dir + item['savename'] + 'epochbar-' + time + '.png'
        plt.tight_layout()
        plt.savefig(figname, dpi=300, bbox_inches='tight', transparent=True)
        plt.show()
        logging.info(f"{item['savename'][:-1]} validation loss plot saved to {figname}")



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


def plot_uq_hist_from_file(fontsize=15, labelsize=13, 
                           datatype: {'absdiff', 'nwaves'} = 'absdiff'):
    """
    Plots the hplus and hcross mismatch uncertainty histograms
    on two subplots with log-scaled x-axes. The function reads 
    mismatch data from a specified CSV file, and creates histograms 
    for the hplus and hcross mismatches.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    if datatype == 'nwaves':
        dir = '../results/20260416/'
        fname = 'uq-hphc-hist-Nwaves-5000-Nruns-100-20260416_083418'
        savename = dir + 'uq-hphc-hist-Nwaves-5000-Nruns-100'
    else:
        dir = '../results/20260401/'
        fname = 'uq-test-hist-mean-abs-diff-5000-20260401_174233'
        savename = dir + 'uq-hphc-hist-mean-abs-diff-5000'
    dfuq = pd.read_csv(dir+fname+'.csv', header=0)
    hpbins = np.logspace(np.log10(dfuq['mmuq_hplus'].min())+1e-10,
                         np.log10(dfuq['mmuq_hplus'].max())+1e-10, 50)
    dfuq.hist('mmuq_hplus', bins=hpbins, ax=axes[0], grid=False, edgecolor='black', color='lightblue')
    dfuq.hist('mmuq_hcross', bins=hpbins, ax=axes[1], grid=False, edgecolor='black', color='lightblue')
    types = ['mmuq_hplus', 'mmuq_hcross']
    titles = [ '$\\mathbf{h_{+}}$', '$\\mathbf{h_{\\times}}$']
    for i in range(len(types)):
        ax = axes[i]
        if datatype == 'absdiff':
            ax.set_xlim(1e-2,1e0)
            xloc, yloc, ha = 0.95, 0.95, 'right'
        else:
            ax.set_xlim(1e-5, 1e-1)
            xloc, yloc, ha = 0.05, 0.95, 'left'
        ax.set_xscale('log')
        ax.tick_params(which="both", direction='in', top=True, right=True)
        ax.tick_params(labelsize=labelsize)
        ax.set_xlabel('Mismatch Standard Deviation', fontsize=fontsize)
        ax.set_ylabel('Count', fontsize=fontsize)
        ax.text(xloc, yloc, titles[i], fontweight='bold',
                transform=ax.transAxes, fontsize=labelsize, va='top', ha=ha)
        ax.text(xloc, yloc - 0.1, f'Mode: {dfuq[types[i]].mode()[0]:.2e}\nMean: {dfuq[types[i]].mean():.2e}\nMedian: {dfuq[types[i]].median():.2e}',
                transform=ax.transAxes, fontsize=labelsize, va='top', ha=ha)
        ax.set_title(None)
        ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
    plt.tight_layout()
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(savename+'-'+now+'.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig(savename+'-white'+'-'+now+'.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    logging.info(f"Uncertainty histograms saved to {dir}")


def plot_timecompare_from_file(fname=None, dir=DIR, time=TIME, fontsize=12, labelsize=10,
                               corrected=True):
    """
    Plot the time taken to generate different variants of the SEOBNRv4 waveform
    versus the base waveform implementation.
    """
    approximant = 'SEOBNRv4'
    dir = '../results/'
    date = '20260401'
    # fname = '20260228/timecomplexity_results-20260228_031118.csv'
    fname = f'{date}/timecomplexity_results-cuda-20260401_012112'
    fname += '-corrected' if corrected else ''
    modeldf = pd.read_csv(dir + fname + '.csv', skiprows=1,
                          names=['Nruns', 'modeltimes'])
    otherdf = pd.read_csv(dir + '20260126/timecomplexity_compare_20260126_031310.csv', skiprows=1,
                          names=['Nruns', 'basetimes', 'romtimes', 'opttimes'])
    print(otherdf.head())
    
    # Print average per waveform generation times for each method
    avg_model_time = modeldf['modeltimes'].mean() / modeldf['Nruns'].mean()
    avg_base_time = otherdf['basetimes'].mean() / otherdf['Nruns'].mean()
    avg_rom_time = otherdf['romtimes'].mean() / otherdf['Nruns'].mean()
    avg_opt_time = otherdf['opttimes'].mean() / otherdf['Nruns'].mean()
    logging.info(f"Average time per waveform generation:")
    logging.info(f"\tModel: {avg_model_time:1e} seconds")
    logging.info(f"\tBase: {avg_base_time:1e} seconds")
    logging.info(f"\tROM: {avg_rom_time:1e} seconds")
    logging.info(f"\tOptimized: {avg_opt_time:1e} seconds")
    # Print how fast ML model is compared to base, ROM, and optimized, at 99 waveforms generated
    n = 99
    # print(modeldf[modeldf['Nruns'] == n]['modeltimes'].values[0])
    # print(otherdf[otherdf['Nruns'] == n]['basetimes'].values)
    model_time_n = modeldf[modeldf['Nruns'] == n]['modeltimes'].values[0] / n
    base_time_n = otherdf[otherdf['Nruns'] == n]['basetimes'].values[0] / n
    rom_time_n = otherdf[otherdf['Nruns'] == n]['romtimes'].values[0] / n
    opt_time_n = otherdf[otherdf['Nruns'] == n]['opttimes'].values[0] / n
    logging.info(f"Time per waveform generation at {n} waveforms:")
    logging.info(f"\tModel: {model_time_n:1e} seconds")
    logging.info(f"\tBase: {base_time_n:1e} seconds")
    logging.info(f"\tROM: {rom_time_n:1e} seconds")
    logging.info(f"\tOptimized: {opt_time_n:1e} seconds")
    logging.info(f"Model is {base_time_n/model_time_n:.2f}x faster than Base, {rom_time_n/model_time_n:.2f}x faster than ROM, and {opt_time_n/model_time_n:.2f}x faster than Optimized at {n} waveforms.")

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
    figname = dir + f'{date}/' + 'timecomplexity-compare-long-gpu-' + datetime.now().strftime('%Y%m%d_%H%M%S')
    figname += '-corrected' if corrected else ''
    plt.savefig(figname+'.png', dpi=300, transparent=True)
    plt.savefig(figname+'-white.png', dpi=300)
    plt.show()
    plt.close()
    logging.info("Time complexity comparison plot saved.")



def read_params_from_file():
    """
    Reads the parameters of training data from a csv file,
    and prints the summary of the parameters in a nice format. 
    The parameters file columns are:
        sample,mass1,mass2,spin1z,spin2z,delta_t,f_lower,sample_rate,padded,truncated
    """
    params_file = '../data/params-SEOBNRv4-train-100000-fcutoff-uniform-aligned-regen.csv'
    df = pd.read_csv(params_file, header=0)
    logging.info(f"Summary of training data parameters from {params_file}:")
    logging.info(f"Total samples: {len(df)}")
    logging.info(f"Mass1: min={df['mass1'].min():.2f}, max={df['mass1'].max():.2f}, mean={df['mass1'].mean():.2f}, median={df['mass1'].median():.2f}")
    logging.info(f"Mass2: min={df['mass2'].min():.2f}, max={df['mass2'].max():.2f}, mean={df['mass2'].mean():.2f}, median={df['mass2'].median():.2f}")
    logging.info(f"Spin1z: min={df['spin1z'].min():.2f}, max={df['spin1z'].max():.2f}, mean={df['spin1z'].mean():.2f}, median={df['spin1z'].median():.2f}")
    logging.info(f"Spin2z: min={df['spin2z'].min():.2f}, max={df['spin2z'].max():.2f}, mean={df['spin2z'].mean():.2f}, median={df['spin2z'].median():.2f}")
    logging.info(f"Delta_t: min={df['delta_t'].min():.4f}, max={df['delta_t'].max():.4f}, mean={df['delta_t'].mean():.4f}, median={df['delta_t'].median():.4f}")
    logging.info(f"F_lower: min={df['f_lower'].min():.2f}, max={df['f_lower'].max():.2f}, mean={df['f_lower'].mean():.2f}, median={df['f_lower'].median():.2f}")
    logging.info(f"Sample_rate: min={df['sample_rate'].min():.2f}, max={df['sample_rate'].max():.2f}, mean={df['sample_rate'].mean():.2f}, median={df['sample_rate'].median():.2f}")
    print(df.head())



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
    # mismatch_anal(args)
    # plot_rom_opt_mm_hist(fontsize=20, labelsize=15)
    # plot_timecompare_from_file()
    # plot_flexcvae_loss(dir=args.dir, time=args.time)
    # plot_uq_hist_from_file(datatype='nwaves')
    # plot_loss_from_file(onlyprintsteps=False)
    read_params_from_file()

