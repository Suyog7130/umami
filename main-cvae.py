"""
2024/12/04 @ Osaka :
This program is to generate GW waveform time series for 
different kinds of approximants, using Conditional 
Autoencoder (CVAE) model from the paper:
https://doi.org/10.1103/PhysRevD.103.124051

TODO
----
1) How to make sure the generated sample length is 
    outputsize ?
3) Make the code robust ??
4) That paper uses ~25000 training samples, so we 
    shouldn't req anymore samples than that!
5) Use `class generator` for sending out batchsize sized 
    chunks of data for training !!

Necessarily need to scale the desired output by e^-20

2025/03/18 @ Monash :
Working on the minimal working example today!
"""

import os
import argparse
import h5py
import numpy as np
import pandas as pd

# import matplotlib
# matplotlib.use('Agg')   # non GUI backend
import matplotlib.pyplot as plt

# from sklearn import metrics
from tqdm import tqdm

import torch
import logging

torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

# from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary

import pycbc.noise
import pycbc
from pycbc.waveform import get_td_waveform
import pycbc.waveform, pycbc.noise, pycbc.psd, pycbc.distributions, \
     pycbc.detector

from sklearn import metrics

from datetime import datetime

import sys
sys.path.append('~Dropbox/plotutils-work/plotutils/')
from plotutils import putils

from datacvae import CustomDataset
from datacvae import PRESET_ARRAY_SIZE, SAMPLE_RATE, DELTA_T, f_lower, sample_len
from cvae import CVAE



# def train_one_epoch(training_loader, epoch_index, tb_writer=None):
#      running_loss = 0.
#      # last_loss = 0.

#      # Here, we use enumerate(training_loader) instead of
#      # iter(training_loader) so that we can track the batch
#      # index and do some intra-epoch reporting
#      # `training_loader` automatically divides data into batches
#      for i, data in enumerate(training_loader):
#           # Every data instance is an input + label pair
#           inputs, labels = data
#           # print('labels=',labels)
#           inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1])
#           labels = labels.reshape(labels.shape[0], 1, labels.shape[1])
#           logging.info(inputs.shape)

#           # Zero your gradients for every batch!
#           optimizer.zero_grad()

#           # Make predictions for this batch
#           outputs = model(inputs)

#           # Compute the loss and its gradients
#           loss = loss_fn(outputs, labels)
#           loss.backward()

#           # Adjust learning weights
#           optimizer.step()

#           last_loss = loss.item()
#           running_loss += last_loss
#           logging.info(f'  batch {i+1} loss: {last_loss}')

#      avg_loss = running_loss / (i + 1)
#      return avg_loss


def plot(arr1, arr2, savename='results/', labels=['', ''], noshow=False):
     savename += '.png'
     fig, axes = plt.subplots(2, 1, figsize=(7, 10))
     for arr, label, ax in zip([arr1, arr2], labels, [axes[0], axes[1]]):
          arr = arr.reshape(arr.shape[1])
          xarr = np.linspace(0, 1, len(arr))
          ismax = np.argmax(arr)
          xarr = xarr - xarr[ismax]
          ax.plot(xarr, arr, label=label)
          ax.legend()
     putils.beautifyPlot(axes, grid=True)
     plt.tight_layout()
     plt.savefig(savename, dpi=300)
     if not noshow:
          plt.show()



def train(args):
    savename = timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # epoch = 0

    # best_vloss = 1_000_000.

    # model.n_resblocks = args.n_resblocks

    # tmasses, vmasses = np.arange(5, 75, 1), np.linspace(5.5, 74.5, 1)

    # training_inputs, training_outputs = get_data(
    #      n_samples=args.nsamples, masses=tmasses,
    #      approximant=args.approximant, paramsonly=True)
    # logging.info(training_inputs.shape)
    # validation_inputs, validation_outputs = get_data(
    #      n_samples=int(0.2 * args.nsamples), masses=vmasses,
    #      approximant=args.approximant, paramsonly=True)

    # training_set = CustomDataset(training_inputs, training_outputs,
    #                                        train_device=device)
    # validation_set = CustomDataset(validation_inputs, validation_outputs,
    #                                          train_device=device)
    logging.info(f"Initialing Data with arguments:\n{args.__dict__}")
    train_set = CustomDataset(forwhat='train', approximant=args.approximant,
                            convert=args.convert, hdf_fname=args.datadir+args.approximant+'-train',)
    valid_set = CustomDataset(forwhat='valid', approximant=args.approximant,
                            convert=args.convert, hdf_fname=args.datadir+args.approximant+'-valid',)
    logging.info(f"Train set size: {len(train_set)}")
    logging.info(f"Validation set size: {len(valid_set)}")

    training_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True)
    logging.info(training_loader.__dict__)
    ntbatches = len(training_loader)
    nvbatches = len(validation_loader)
    logging.info(f'Number of Training batches: {ntbatches}')  # this doesn't return the batchsize!
    logging.info(f'Number of Validationg batches: {nvbatches}')

    # Initialize Model
    # `num_classes` is the size of the labels.
    model = CVAE(input_shape=(2, PRESET_ARRAY_SIZE), num_classes=2, key_shape=(2,2)).to(args.device)
    # Add a learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    logging.info('Model Initialized')
    # Scheduler will adjust learning rate after every epoch

    logging.info(f'Starting Training with: {args}')
    train_rloss, valid_rloss = [], []  # running loss every batch
    train_loss, valid_loss = [], []
    netreconloss, netklloss = [], []
    for epoch in tqdm(range(args.epochs), desc='Epoch'):
        model.train(True)
        # avg_loss = train_one_epoch(training_loader, epoch)

        # Train model for one Epoch
        for x, labels, keys in tqdm(training_loader, desc='train-batch'):
            """
            `x` is [freq, amp], `labels` is [m1,m2] and
            `keys` is [[amp-mean,amp-var],[freq-mean,freq-var]]
            """
            x, labels, keys = x.to(args.device), labels.to(args.device), keys.to(args.device)
            optimizer.zero_grad()
            x_recon, zvars = model(x, labels, keys)
            loss, reconloss, klloss = model.loss_function(x, x_recon, zvars)
            loss.backward()
            optimizer.step()
            train_rloss.append(loss.item())
            netreconloss.append(reconloss.item())
            netklloss.append(klloss.item())
        train_loss.append(loss.item())

        model.eval()
        with torch.no_grad():  # Disable gradient computation for validation
            # just reconstruct target and calculate diff
            for vx, vlabels, vkeys in tqdm(validation_loader, desc='val-batch'):
                vx, vlabels, vkeys = vx.to(args.device), vlabels.to(args.device), vkeys.to(args.device)
                vx_recon, vzvars = model(vx, vlabels, vkeys)
                vloss, _reconloss, _klloss = model.loss_function(vx, vx_recon, vzvars)
                valid_rloss.append(vloss.item())
            valid_loss.append(vloss.item())
        tqdm.write(f'Epoch {epoch+1} : train loss {loss.item()} & valid loss {vloss.item()}')
        scheduler.step()  # Update learning rate
        logging.debug(f"x shape: {x.shape}, labels shape: {labels.shape}, keys shape: {keys.shape}")
    
    savename = timestamp + '-' + str(args.epochs)
    if not os.path.isdir('trained-models/'):
        os.makedirs('trained-models/')
    model_path = (f'trained-models/model-{savename}')
    if not args.nosave:
        if not os.path.isdir('results/'):
            os.makedirs('results/')
        torch.save(model.state_dict(), model_path)
        # Save losses to files
        # When using pandas, all arrays should have same length
        dfepoch = pd.DataFrame({
            'train_loss': train_loss,
            'valid_loss': valid_loss,})
        dfepoch.to_csv(f'results/epoch-loss-{timestamp}.csv', index=False)
        np.savetxt(f'results/train-rloss-{timestamp}.txt', train_rloss)
        np.savetxt(f'results/valid-rloss-{timestamp}.txt', valid_rloss)
        dfnet = pd.DataFrame({
            'netreconloss': netreconloss,
            'netklloss': netklloss})
        dfnet.to_csv(f'results/net-loss-{timestamp}.csv', index=False)


    fig, axes = plt.subplots(2, 1, figsize=(5, 10))
    axes[0].plot(np.arange(args.epochs), train_loss, label='training loss')
    axes[0].plot(np.arange(args.epochs), valid_loss, label='validation loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend()
    axes[1].plot(np.arange(args.epochs*ntbatches), train_rloss, label='train running loss')
    axes[1].plot(np.arange(args.epochs*nvbatches), valid_rloss, label='valid running loss')
    axes[1].plot(np.arange(args.epochs*ntbatches), netreconloss, label='reconstruction loss')
    axes[1].plot(np.arange(args.epochs*ntbatches), netklloss, label='latent loss')
    axes[1].set_xlabel('Batch', fontsize=12)
    axes[1].set_yscale('log')  # Set y-axis to logarithmic scale
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend()
    # putils.beautifyPlot(axes)
    plt.tight_layout()
    savepath = 'results/'
    if not os.path.isdir(savepath):
        savepath = '.'
    if not args.nosave:
        plt.savefig(savepath+'/epoch-loss-'+timestamp+'.png', dpi=300)

    # if device != 'cpu':
    #      vlabels = vlabels.to('cpu')
    #      voutputs = voutputs.to('cpu')
    # plot(vlabels[len(vlabels) // 2], voutputs[len(voutputs) // 2],
    #      savename='validation-check-' + savename,
    #      labels=['correct', 'generated'], noshow=args.noshow)
    # df = pd.DataFrame()
    # df['train_loss'] = train_loss
    # df['valid_loss'] = valid_loss
    # df.to_csv(f'results/epoch-loss-{savename}.dat')
    # df1 = pd.DataFrame({'eg_target': np.array(vlabels[len(vlabels) // 2]).flatten(),
    #                          'eg_output': np.array(voutputs[len(voutputs) // 2]).flatten()})
    # df1.to_csv(f'results/output-comparison-{savename}.dat')
    logging.info('Program Successfully Ran !')



class Test:

    def __init__(self, args):
        # for arg in args.__dict__:
        #     setattr(self, arg, args.__dict__[arg])
        self.approximant = args.approximant
        self.convert = args.convert
        self.datadir = args.datadir

        self.batch_size = args.batch_size
        self.noshow = args.noshow
        self.nosave = args.nosave
        self.savedir = '../results/'
        self.test_loader = self.setdataloader()
        logging.info('Test DataLoader set up.')
        self.epochs = 1
        self.model_path = args.model

    def setdataloader(self):
        """
        Set up the DataLoader for the test dataset.
        """
        test_set = CustomDataset(forwhat='test', approximant=self.approximant,
                                convert=self.convert, hdf_fname=self.datadir+self.approximant+'-test',
                                returnattr=True, train_device=args.device, )
        logging.info(f'Reading test data from {self.datadir+self.approximant+"-test.hdf"}')
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=True,
                                 collate_fn=test_set.collate_fn)
        logging.info(f'Test set size: {len(test_set)}')
        return test_loader

    def test(self):
        """
        Test the trained CVAE model using only labels as input.
        TODO: Create a test dir in results in dir and a subfolder with timestamp !!
        """
        logging.info(f"Testing with model: {self.model_path}")
        # Load the trained model
        model = CVAE(input_shape=(2, PRESET_ARRAY_SIZE), num_classes=2, 
                    key_shape=(2,2)).to(args.device)
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        model.to(device)
        model.eval()
        logging.info("Model loaded and set to evaluation mode.")

        for _ in range(self.epochs):
            # `next(iter(self.test_loader))` gives us a batch of data!
            # Thus, `shape(x)` is (batch_size, 2, PRESET_ARRAY_SIZE) etc.
            x, labels, keys, attr, phase = next(iter(self.test_loader))
            logging.debug(attr)

            # plt.plot(range(len(x[0][0])), x[0][0].cpu().numpy(), label='input')
            # if not self.noshow:
            #     plt.show()
            
            # Move labels to the appropriate device
            labels = labels.to(device)

            # Generate reconstructed data
            with torch.no_grad():
                # Use the label-conditioned encoders and decoder to generate data
                z1_mean, z1_log_var = model.encode_label_for_x(labels)
                z1p_mean, z1p_log_var = model.encode_label_for_key(labels)
                z1 = model.reparameterize(z1_mean, z1_log_var)
                z1p = model.reparameterize(z1p_mean, z1p_log_var)
                reconst = model.decode(z1, z1p, labels)
            logging.debug(x.shape, reconst.shape, keys.shape)
            logging.info('Test for current epoch completed. Removing zero padding if any.')
            x, reconst = removezeros(x, reconst, attr)
            logging.info(f'Removed zero padding from input and reconstructed data.')
            logging.info(f'new shapes, Input: {x.shape}, Reconstructed: {reconst.shape}')
            # plot_reconstruct_data(reconst, labels, keys,
            #                       savename=None if self.nosave else self.savedir+'/reconst')
            plot_overplot(x, reconst, labels, keys, 
                          savename=None if self.nosave else self.savedir+'overplot')
            plot_mismatch(x, reconst, labels, keys)
            plot_polarization_mismatch(x, reconst, labels, keys, phase)


def removezeros(x, reconst, attr):
    """
    Remove zero padding from the input and reconstructed data.
    This is useful for visualizing the actual waveform data without padding.

    TODO: If it was truncated, then we don't have to do anything basically!

    Parameters:
    -----------
    x : torch.Tensor
        Original data input to the CVAE.
        Shape is expected to be (batch_size, 2, PRESET_ARRAY_SIZE).
    reconst : torch.Tensor
        Reconstructed data generated by the CVAE.
        Shape is expected to be (batch_size, 2, PRESET_ARRAY_SIZE).
    attr : dict
        Attributes containing information about truncation and padding.

    Returns:
    --------
    x : torch.Tensor
        Original data without zero padding.
    reconst : torch.Tensor
        Reconstructed data without zero padding.
    """
    # if attr['truncated']:
    # nothing to do for truncation, as we assume the data is already truncated.
    #     print(x.shape, reconst.shape, attr['truncated_len'])
    #     x = x[:, :, :attr['truncated_len']]
    #     reconst = reconst[:, :, :attr['truncated_len']]
    if attr['padded']:
        # padding is always at the end of the data.
        x = x[:, :, :attr['padded_at']]
        reconst = reconst[:, :, :attr['padded_at']]
    return x, reconst

def plot_reconstruct_data(reconst, labels, keys, savename='../results/reconst'):
    """
    Visualize the reconstructed data generated by the CVAE.
    Assuming that the Freq-Amp conversion was done before training.
    Plots two panel with, say three, random samples of the reconstructed data.
    The panels are for Amplitude and Frequency, respectively.

    Parameters:
    -----------
    reconst : torch.Tensor
        Reconstructed data generated by the CVAE.
    labels : torch.Tensor
        Labels associated with the reconstructed data.
    save_path : str, optional
        Path to save the visualization plot. If None, the plot is not saved.

    Returns:
    --------
    None

    Outputs:
    -------
    Displays a plot of the reconstructed data.
    """
    reconst = reconst.cpu().numpy()
    # print(reconst.shape)
    labels = labels.cpu().numpy()
    # print(labels.shape)
    keys = keys.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for j in range(3):
        i = np.random.randint(0, 49, size=1)
        data = reconst[i].reshape([2,PRESET_ARRAY_SIZE])
        amp, freq = data[0], data[1]
        # De-normalize the reconstructed data!
        key = keys[i].reshape([2,2])
        amp_mean, amp_std = key[0][0], key[0][1]
        freq_mean, freq_std = key[1][0], key[1][1]
        amp = (amp * amp_std) + amp_mean
        freq = (freq * freq_std) + freq_mean
        axes[0].plot(np.arange(len(amp)), amp, '-', label=f"Label: {labels[i]}")
        axes[1].plot(np.arange(len(freq)), freq, '-', label=f"Label: {labels[i]}")
        # axes[0].set_title(f"Reconstructed (Label: {labels[i]})")
        # axes[j].axis('off')

    for i, axlabel in enumerate(['Amplitude', 'Frequency']):
        axes[i].set_xlabel('Sample length', fontsize=12)
        axes[i].set_ylabel(axlabel, fontsize=12)
        axes[i].set_title(f'Reconstructed {axlabel}', fontsize=12)
        axes[i].legend()

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    putils.beautifyPlot(axes)
    if savename:
        savename += '-' + datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(savename+'.png', dpi=300)
        print(f"Reconstruction plot saved to {savename}")
    plt.show()


def plot_overplot(x, reconst, labels, keys, savename='../results/overplot',
                  reshape2orig=False):
    """
    Overplot the original data and the reconstructed data from the CVAE.
    Plots two panel with, say three, random samples of the original and 
    reconstructed data. The panels are for Amplitude and Frequency, respectively.

    Parameters:
    -----------
    x : torch.Tensor
        Original data input to the CVAE.
    reconst : torch.Tensor
        Reconstructed data generated by the CVAE.
    labels : torch.Tensor
        Labels associated with the original data.
    keys : torch.Tensor
        Keys associated with the original data.

    Returns:
    --------
    None

    Outputs:
    -------
    Displays a plot of the original and reconstructed data.
    """
    # Change tensors to numpy arrays for plotting
    x = x.cpu().numpy()
    reconst = reconst.cpu().numpy()
    labels = labels.cpu().numpy()
    keys = keys.cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    for j in range(1):
        i = np.random.randint(0, 49, size=1)
        if reshape2orig:
            # Reshape to original data shape
            orig_data = x[i].reshape([2,PRESET_ARRAY_SIZE])
            reconst = reconst.reshape([2,PRESET_ARRAY_SIZE])
        else:
            # Use the original shape of the data
            logging.debug(x.shape, reconst.shape)
            orig_data = x[i].reshape([2,x.shape[2]])
            recon_data = reconst[i].reshape([2,reconst.shape[2]])
        
        orig_amp, orig_freq = orig_data[0], orig_data[1]
        recon_amp, recon_freq = recon_data[0], recon_data[1]
        
        # Obtain the keys for normalization
        key = keys[i].reshape([2,2])
        amp_mean, amp_std = key[0][0], key[0][1]
        freq_mean, freq_std = key[1][0], key[1][1]

        # # De-normalize the original data!
        # orig_amp = (orig_amp * amp_std) + amp_mean
        # orig_freq = (orig_freq * freq_std) + freq_mean
        # # De-normalize the reconstructed data!
        # recon_amp = (recon_amp * amp_std) + amp_mean
        # recon_freq = (recon_freq * freq_std) + freq_mean
        
        axes[0].plot(np.arange(len(orig_amp)), orig_amp, '-', label=f"Original")
        axes[0].plot(np.arange(len(recon_amp)), recon_amp, '--', label=f"Reconstructed")
        
        axes[1].plot(np.arange(len(orig_freq)), orig_freq, '-', label=f"Original")
        axes[1].plot(np.arange(len(recon_freq)), recon_freq, '--', label=f"Reconstructed")

    for i, axlabel in enumerate(['Amplitude', 'Frequency']):
        axes[i].set_xlabel('Sample length', fontsize=12)
        axes[i].set_ylabel(axlabel, fontsize=12)
        axes[i].set_title(f'Overplot {axlabel}', fontsize=12)
        axes[i].legend(title=f'({float(labels[i][0]), float(labels[i][1])})', fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    putils.beautifyPlot(axes)
    if savename:
        savename += '-' + datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(savename+'.png', dpi=300)
        print(f"Overplot saved to {savename}")
    plt.show()


def calculate_mismatch(target, reconstructed):
    """
    Calculate the normalized mismatch between the target and reconstructed waveforms.
    Mismatch = 1 - ( <a|b> / sqrt(<a|a> + <b|b>) )
    where <a|b> is the inner product (dot product).

    Parameters:
    -----------
    target : np.ndarray or torch.Tensor
        The correct/original waveform, shape (..., N)
    reconstructed : np.ndarray or torch.Tensor
        The reconstructed waveform, shape (..., N)

    Returns:
    --------
    mismatch : float or np.ndarray
        The mismatch value(s), 0 means perfect match, 1 means orthogonal.
    """
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()
    # TODO: use PSD to whiten the recombined waveform and then calculate the mismatch.
    # calculate the fitting factor
    inner_product = np.sum(target * reconstructed, axis=-1)
    target_norm = np.sqrt(np.sum(target * target, axis=-1))
    reconstructed_norm = np.sqrt(np.sum(reconstructed * reconstructed, axis=-1))
    # Calculate the match
    match = inner_product / (target_norm * reconstructed_norm)
    mismatch = 1 - match
    return mismatch

def plot_mismatch(x, reconst, labels, keys,
                  savedir='../results/', reshape2orig=False):
    """
    Plot the mismatch between the original and reconstructed data.

    Parameters:
    -----------
    x : torch.Tensor
        Original data input to the CVAE.
    reconst : torch.Tensor
        Reconstructed data generated by the CVAE.
    labels : torch.Tensor
        Labels associated with the original data.
    keys : torch.Tensor
        Keys associated with the original data.
    reshape2orig : bool, optional
        If True, reshapes the data to the original shape before calculating mismatch.

    Returns:
    --------
    None

    Outputs:
    -------
    Displays a plot of the mismatch values.
    """
    x = x.cpu().numpy()
    reconst = reconst.cpu().numpy()
    labels = labels.cpu().numpy()
    keys = keys.cpu().numpy()
    
    chirpmasses = np.zeros((labels.shape[0], 1))  # Store chirp masses for each sample
    totalmasses = np.zeros((labels.shape[0], 1))  # Store total masses for each sample
    massratios = np.zeros((labels.shape[0], 1))  # Store mass ratios for each sample
    mismatch_amp = np.zeros((x.shape[0], 1))
    mismatch_freq = np.zeros((x.shape[0], 1))

    # TODO: Maybe remove this `for` loop and use vectorized operations?
    for i in range(x.shape[0]):
        # i = np.random.randint(0, 49, size=1)
        if reshape2orig:
            # Reshape to original data shape
            orig_data = x[i].reshape([2,PRESET_ARRAY_SIZE])
            reconst = reconst.reshape([2,PRESET_ARRAY_SIZE])
        else:
            # Use the original shape of the data
            logging.debug(x.shape, reconst.shape)
            orig_data = x[i].reshape([2,x.shape[2]])
            recon_data = reconst[i].reshape([2,reconst.shape[2]])
        
        orig_amp, orig_freq = orig_data[0], orig_data[1]
        recon_amp, recon_freq = recon_data[0], recon_data[1]
        
        # Obtain the keys for normalization
        key = keys[i].reshape([2,2])
        amp_mean, amp_std = key[0][0], key[0][1]
        freq_mean, freq_std = key[1][0], key[1][1]

        # De-normalize the original data!
        orig_amp = (orig_amp * amp_std) + amp_mean
        orig_freq = (orig_freq * freq_std) + freq_mean
        # De-normalize the reconstructed data!
        recon_amp = (recon_amp * amp_std) + amp_mean
        recon_freq = (recon_freq * freq_std) + freq_mean
        
        # Calculate mismatch
        mismatch_amp[i] = calculate_mismatch(orig_amp, recon_amp)
        mismatch_freq[i] = calculate_mismatch(orig_freq, recon_freq)
        logging.debug(f"Mismatch for Amplitude: {mismatch_amp[i]}, Frequency: {mismatch_freq[i]}")

        # Calculate chirp mass
        m1, m2 = labels[i][0], labels[i][1]
        chirp_mass = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
        logging.debug(f"Chirp mass for sample {i}: {chirp_mass}")
        chirpmasses[i] = chirp_mass
        totalmasses[i] = m1 + m2
        massratios[i] = m1 / m2

    for massarr, xname in zip([chirpmasses, totalmasses, massratios],
                               ['Chirp Mass', 'Total Mass', 'Mass Ratio']):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(massarr, mismatch_amp, '.', label=f'Amplitude',
                markeredgewidth=0.75, alpha=0.75)
        ax.plot(massarr, mismatch_freq, '.', label=f'Frequency',
                markeredgewidth=0.75, alpha=0.75)
        ax.set_xlabel(xname, fontsize=12)
        ax.set_ylabel('Mismatch', fontsize=12)
        ax.set_yscale('log')  # Set y-axis to logarithmic scale
        # show minor ticks on x-axis
        # ax.xaxis.set_minor_locator(plt.AutoLocator())
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        plt.legend()
        # plt.tight_layout()
        # putils.beautifyPlot([ax])
        savename = 'mismatch-'+xname.replace(' ','')+ '-' + datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(savedir+savename+'.png', dpi=300, bbox_inches='tight')
        logging.info(f"Mismatch plot saved to {savedir+savename}.png")
        plt.close()


def calc_polarization_mismatch(hp_orig, hp_recon):
    """
    Calculate the mismatch between the original and reconstructed hplus/hcross waveforms.

    Parameters:
    -----------
    hp_orig : np.ndarray or torch.Tensor
        The original hplus waveform.
    hp_recon : np.ndarray or torch.Tensor
        The reconstructed hplus waveform.

    Returns:
    --------
    mismatch : float
        The mismatch value, 0 means perfect match, 1 means orthogonal.
    """
    if isinstance(hp_orig, torch.Tensor):
        hp_orig = hp_orig.detach().cpu().numpy()
    if isinstance(hp_recon, torch.Tensor):
        hp_recon = hp_recon.detach().cpu().numpy()
    
    from pycbc.filter import match as matchfunc
    from pycbc.psd import aLIGOZeroDetHighPower
    psd = aLIGOZeroDetHighPower(length=sample_len,
                                delta_f=1.0/hp_recon.duration,
                                low_freq_cutoff=f_lower)
    match = matchfunc(hp_orig, hp_recon, psd=psd, low_frequency_cutoff=f_lower)
    mismatch = 1 - match
    return mismatch

def phase_from_frequency(freq, dt, theta0=0.0):
    """
    Compute gravitational-wave phase from a frequency time series.

    Parameters
    ----------
    freq : array_like
        Instantaneous frequency time series (Hz).
    dt : float
        Time step between samples (seconds).
    phi0 : float, optional
        Initial phase (radians). Default is 0.
        
    Returns
    -------
    phase : ndarray
        Phase time series (radians).
    """
    from scipy.integrate import cumulative_trapezoid
    # Integrate frequency using trapezoidal rule
    theta_integral = cumulative_trapezoid(freq, dx=dt, initial=0.0)
    # Multiply by 2π and add initial phase
    return theta0 + 2 * np.pi * theta_integral


def polarizations_from_ampfreq(amp, freq, orig_phase=None):
    """
    Convert amplitude and frequency to hplus and hcross polarizations.
    """
    phase = phase_from_frequency(freq, dt=1.0/SAMPLE_RATE)
    hplus = amp * np.cos(phase)
    hcross = amp * np.sin(phase)
    return hplus, hcross

def plot_polarization_mismatch(x, reconst, labels, keys, phase,
                                savedir='../results/', reshape2orig=False):
    """
    Plot the mismatch between the original and reconstructed hplus/hcross waveforms.

    Parameters:
    -----------
    x : torch.Tensor
        Original data input to the CVAE.
    reconst : torch.Tensor
        Reconstructed data generated by the CVAE.
    labels : torch.Tensor
        Labels associated with the original data.
    keys : torch.Tensor
        Keys associated with the original data.
    phase : torch.Tensor
        Phase information associated with the original data.

    Returns:
    --------
    None

    Outputs:
    -------
    Displays a plot of the polarization mismatch values.
    """
    x = x.cpu().numpy()
    reconst = reconst.cpu().numpy()
    labels = labels.cpu().numpy()
    keys = keys.cpu().numpy()
    phase = phase.cpu().numpy()

    chirpmasses = np.zeros((labels.shape[0], 1))  # Store chirp masses for each sample
    totalmasses = np.zeros((labels.shape[0], 1))  # Store total masses for each sample
    massratios = np.zeros((labels.shape[0], 1))  # Store mass ratios for each sample
    mismatch_hplus = np.zeros((x.shape[0], 1))
    mismatch_hcross = np.zeros((x.shape[0], 1))

    for i in range(x.shape[0]):
        if reshape2orig:
            # Reshape to original data shape
            orig_data = x[i].reshape([2,PRESET_ARRAY_SIZE])
            reconst = reconst.reshape([2,PRESET_ARRAY_SIZE])
        else:
            # Use the original shape of the data
            logging.debug(x.shape, reconst.shape)
            orig_data = x[i].reshape([2,x.shape[2]])
            recon_data = reconst[i].reshape([2,reconst.shape[2]])

        orig_amp, orig_freq = orig_data[0], orig_data[1]
        recon_amp, recon_freq = recon_data[0], recon_data[1]
        
        # Obtain the keys for normalization
        key = keys[i].reshape([2,2])
        amp_mean, amp_std = key[0][0], key[0][1]
        freq_mean, freq_std = key[1][0], key[1][1]

        # De-normalize the original data!
        orig_amp = (orig_amp * amp_std) + amp_mean
        orig_freq = (orig_freq * freq_std) + freq_mean
        # De-normalize the reconstructed data!
        recon_amp = (recon_amp * amp_std) + amp_mean
        recon_freq = (recon_freq * freq_std) + freq_mean
        
        # Combine original Amp/Freq to hplus/hcross
        hp_orig = orig_amp * np.cos(phase)  # this is original phase
        hc_orig = orig_amp * np.sin(phase)

        # Calculate hplus/hcross for reconstructed data
        hp_recon, hc_recon = polarizations_from_ampfreq(recon_amp, recon_freq, phase)

        # Calculate mismatch for hplus and hcross
        mismatch_hplus[i] = calc_polarization_mismatch(hp_orig, hp_recon)
        mismatch_hcross[i] = calc_polarization_mismatch(hc_orig, hc_recon)
        logging.info(f"Mismatch for hplus: {mismatch_hplus[i]}, hcross: {mismatch_hcross[i]}")

        # Calculate chirp mass
        m1, m2 = labels[i][0], labels[i][1]
        chirp_mass = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
        logging.debug(f"Chirp mass for sample {i}: {chirp_mass}")
        chirpmasses[i] = chirp_mass
        totalmasses[i] = m1 + m2
        massratios[i] = m1 / m2

    for massarr, xname in zip([chirpmasses, totalmasses, massratios],
                               ['Chirp Mass', 'Total Mass', 'Mass Ratio']):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(massarr, mismatch_hplus, '.', label='$h_{+}$',
                markeredgewidth=0.75, alpha=0.75)
        ax.plot(massarr, mismatch_hcross, '.', label='$h_{\\times}$',
                markeredgewidth=0.75, alpha=0.75)
        ax.set_xlabel(xname, fontsize=12)
        ax.set_ylabel('Mismatch', fontsize=12)
        ax.set_yscale('log')  # Set y-axis to logarithmic scale
        # show minor ticks on x-axis
        # ax.xaxis.set_minor_locator(plt.AutoLocator())
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        plt.legend()
        # plt.tight_layout()
        # putils.beautifyPlot([ax])
        savename = 'mismatch-hphc-'+xname.replace(' ','')+ '-' + datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(savedir+savename+'.png', dpi=300, bbox_inches='tight')
        logging.info(f"Mismatch plot saved to {savedir+savename}.png")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Non-Eccentric GW Generator')

    parser.add_argument('--nsamples', action='store', default=1000, type=int,
                            help='default=%(default)s')
    parser.add_argument('--approximant', action='store', default='IMRPhenomD', )
    parser.add_argument('--batch-size', action='store', default=50, type=int,
                            help='default=%(default)s')
    parser.add_argument('--epochs', action='store', default=10, type=int,
                            help='default=%(default)s')
    parser.add_argument('--n-resblocks', action='store', default=16, type=int,
                            help='default=%(default)s')
    parser.add_argument('--convert', action='store_true', default=False,
                    help='Convert the strain to phase and amplitude')

    parser.add_argument('--validation-split', action='store', default=0.2, type=float,
                            help='no. of validation samples = validation_split * nsamples \
                            (dafault=%(default)s)')
    parser.add_argument('--sample-rate', action='store', default=2048.0, type=float,
                            help='default=%(default)s')
    parser.add_argument('--sample-duration', action='store', default=1.0, type=float,
                            help='default=%(default)s sec')
    
    parser.add_argument('--datadir', action='store', default='../data/',
                            help='directory where data is stored (default=%(default)s)')

    parser.add_argument('--test', action='store_true', default=False,
                            help='whether to test?')
    parser.add_argument('--model', action='store', default='../trained-models/model-20250526_070915-1',
                        help='path to already trained model.')

    parser.add_argument('--noshow', action='store_true', default=False,
                            help='Do not show output Plot !')
    parser.add_argument('--nosave', action='store_true', default=False,
                            help='Do not save output files and plots!')

    parser.add_argument('--verbose', '-v', action='store_true', help="Print update messages.")
    parser.add_argument('--debug', action='store_true', help="Show debug messages.")

    args = parser.parse_args()

    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        print('Verbose mode!')
        log_level = logging.INFO
    else:
        log_level = logging.WARN

    # Need to use `force` here to override default `logging` settings
    logging.basicConfig(format='%(levelname)s | %(asctime)s: %(message)s',
                            level=log_level, datefmt='%y-%m-%d %H:%M:%S',
                            force=True)
    
    # TODO: Sometimes, `cuda` is not available, and `torch.cuda.is_available()` just goes dead,
    # with no error message. This last happended on 2025-07-22, right during and after a maintanence!
    # device = (
    #     "cuda"
    #     if torch.cuda.is_available()
    #     else "mps"
    #     if torch.backends.mps.is_available()
    #     else "cpu"
    # )
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    args.device = device
    logging.info(f"using {device} device !")

    if args.test:
        Test(args).test()
    else:
        # Always reads data from HDF file now!!
        train(args)
