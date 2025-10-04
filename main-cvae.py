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
import matplotlib.ticker as tck

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
    # make save directory
    today = datetime.today().strftime('%Y%m%d')
    if not os.path.isdir(f'../results/{today}/'):
        os.makedirs(f'../results/{today}/')
    savedir = f'../results/{today}/'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

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

    num_classes = 2  # m1 and m2
    logging.info(f"Initialing Data with arguments:\n{args.__dict__}")
    trainhdf = args.datadir + args.approximant + '-train'
    validhdf = args.datadir + args.approximant + '-val'
    if args.fcutoff:
        trainhdf += '-f_cutoff'
        validhdf += '-f_cutoff'
    if args.aligned:
        num_classes = 4  # m1, m2, spin1z, spin2z
        trainhdf += '-100-fcutoff-uniform-aligned'
        validhdf += '-100-fcutoff-uniform-aligned'

    logging.info(f'Reading training data from {trainhdf}.hdf')
    train_set = CustomDataset(forwhat='train', approximant=args.approximant,
                            convert=args.convert, hdf_fname=trainhdf,)
    logging.info(f'Reading validation data from {validhdf}.hdf')
    valid_set = CustomDataset(forwhat='valid', approximant=args.approximant,
                            convert=args.convert, hdf_fname=validhdf,)
    # logging.info(f"Train set size: {len(train_set)}")
    # logging.info(f"Validation set size: {len(valid_set)}")

    training_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True)
    logging.info(training_loader.__dict__)
    ntbatches = len(training_loader)
    nvbatches = len(validation_loader)
    logging.info(f'Number of Training batches: {ntbatches}')  # this doesn't return the batchsize!
    logging.info(f'Number of Validationg batches: {nvbatches}')

    # Initialize Model
    # `num_classes` is the size of the labels.
    if args.fcutoff or args.aligned:
        PRESET_ARRAY_SIZE = 8190
    model = CVAE(input_shape=(2, PRESET_ARRAY_SIZE), num_classes=num_classes, key_shape=(2,2)).to(args.device)
    # Add a learning rate scheduler
    # Scheduler will adjust learning rate after every epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    logging.info('Model Initialized')

    logging.info(f'Starting Training with: {args}')
    train_rloss, valid_rloss = [], []  # running loss every batch
    train_loss, valid_loss = [], [] 
    netreconloss, netklloss = [], []
    for epoch in tqdm(range(args.epochs), desc='Epoch'):
        model.train(True)
        # avg_loss = train_one_epoch(training_loader, epoch)

        # Train model for one Epoch
        for x, labels, keys in tqdm(training_loader, total=len(training_loader),
                                    desc='batch'):
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
    if not os.path.isdir('../trained-models/'):
        os.makedirs('../trained-models/')
    model_path = f'../trained-models/model-{savename}'
    if not args.nosave:
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        torch.save(model.state_dict(), model_path)
        # Save losses to files
        # When using pandas, all arrays should have same length
        dfepoch = pd.DataFrame({
            'train_loss': train_loss,
            'valid_loss': valid_loss,})
        dfepoch.to_csv(savedir + f'epoch-loss-{timestamp}.csv', index=False)
        np.savetxt(savedir + f'train-rloss-{timestamp}.txt', train_rloss)
        np.savetxt(savedir + f'valid-rloss-{timestamp}.txt', valid_rloss)
        dfnet = pd.DataFrame({
            'netreconloss': netreconloss,
            'netklloss': netklloss})
        dfnet.to_csv(savedir + f'net-loss-{timestamp}.csv', index=False)

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
    if not args.nosave:
        plt.savefig(savedir + f'epoch-loss-{timestamp}.png', dpi=300)

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
        super().__init__()
        # for arg in args.__dict__:
        #     setattr(self, arg, args.__dict__[arg])
        self.approximant = args.approximant
        self.convert = args.convert
        self.datadir = args.datadir
        self.testhdf = self.datadir+self.approximant+'-test'
        if args.fcutoff:
            self.testhdf += '-f_cutoff'
        self.batch_size = args.batch_size
        self.test_loader = self.setdataloader()

        self.noshow = args.noshow
        self.nosave = args.nosave

        today = datetime.today().strftime('%Y%m%d')
        if not os.path.isdir(f'../results/{today}/'):
            os.makedirs(f'../results/{today}/')
        self.savedir = f'../results/{today}/'

        self.epochs = 1
        self.model_path = '../trained-models/' + args.model
        logging.info('Test DataLoader set up.')

    def setdataloader(self, batch_size=None, custom_batch=None):
        """
        Set up the DataLoader for the test dataset.
        """
        batch_size = batch_size if batch_size is not None else self.batch_size
        test_set = CustomDataset(forwhat='test', approximant=self.approximant,
                                convert=self.convert, hdf_fname=self.testhdf,
                                returnattr=True, train_device=args.device, 
                                custom_batch=custom_batch)
        logging.info(f'Reading test data from {self.testhdf}')
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True,
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
        preset_array_size = 8190 if args.fcutoff else PRESET_ARRAY_SIZE
        model = CVAE(input_shape=(2, preset_array_size), num_classes=2, 
                    key_shape=(2,2)).to(args.device)
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        model.to(device)
        model.eval()
        logging.info("Model loaded and set to evaluation mode.")

        # Initialize dataframe to store mismatch results of whole test set!
        dfmm = pd.DataFrame(columns=['chirp_mass', 'total_mass', 'mass_ratio',
                                     'mismatch_amp', 'mismatch_freq', 
                                     'mismatch_hplus', 'mismatch_hcross',])

        # for _ in range(self.epochs):
            # `next(iter(self.test_loader))` gives us a batch of data!
            # Thus, `shape(x)` is (batch_size, 2, PRESET_ARRAY_SIZE) etc.

        # Iterate over all the batches
        num_saved_overplots = 0
        for (x, labels, keys, phases, attr) in tqdm(iter(self.test_loader)):
            # logging.debug(f"Attributes: {attr}")  # Ensure 'attr' is defined or replace with the correct variable

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
            logging.debug(f"x shape: {x.shape}, reconst shape: {reconst.shape}, keys shape: {keys.shape}")
            logging.info('Test for current batch completed. Removing zero padding if any.')
            x, reconst, phases = removezeros(x, reconst, phases, attr)
            logging.debug(f'new shapes, Input: {x.shape}, Reconstructed: {reconst.shape}, phases: {phases.shape}')

            # Save 5-10 example reconstructed and overplot figures
            if num_saved_overplots < 10:
                # plot_reconstruct_data(reconst, labels, keys,
                #                       savename=None if self.nosave else self.savedir+'/reconst')
                plot_overplot(x, reconst, labels, keys,
                              savename=None if self.nosave else self.savedir+'overplot')
                num_saved_overplots += 1

            # TODO: How do we know that the massratio arrays etc. correspond to correct values
            #       in the mismatch arrays ?? Well, since the `x` and `reconst` are the same for
            #       all the samples in the batch, we can just use the first sample's values, right?
            logging.info("Calculating Amp/Freq mismatch for current batch.")
            mismatch_amp, mismatch_freq, chirpmasses, totalmasses, massratios \
                = plot_mismatch(x, reconst, labels, keys, savedir=self.savedir, nobatchwiseplot=True)
            logging.info("Amplitude and Frequency mismatch calculated for current batch.")
            logging.info("Calculating hplus/hcross mismatch for current batch.")
            mismatch_hplus, mismatch_hcross, chirpmasses, totalmasses, massratios, num_saved_overplots \
                = plot_polarization_mismatch(x, reconst, labels, keys, phases, 
                                             savedir=self.savedir, nobatchwiseplot=True,
                                             num_saved_overplots=num_saved_overplots)
            logging.info("hplus/hcross mismatch calculated for current batch.")
            logging.info(f"Tests completed for current batch.")
            # Save the mismatch results to the dataframe
            dfmm = pd.concat([dfmm, pd.DataFrame({
                'chirp_mass': chirpmasses.flatten(),
                'total_mass': totalmasses.flatten(),
                'mass_ratio': massratios.flatten(),
                'mismatch_amp': mismatch_amp.flatten(),
                'mismatch_freq': mismatch_freq.flatten(),
                'mismatch_hplus': mismatch_hplus.flatten(),
                'mismatch_hcross': mismatch_hcross.flatten(),
            })], ignore_index=True)
            logging.info(f"Batch mismatch results appended to dataframe. Current size: {dfmm.shape}")
        logging.info("All test batches completed.")

        # Plot the mismatch results for the entire test set using the dataframe
        for (massarr, xname) in zip(
            [dfmm['chirp_mass'], dfmm['total_mass'], dfmm['mass_ratio']],
            ['Chirp Mass', 'Total Mass', 'Mass Ratio']):
            logging.info(f"Plotting mismatch vs {xname} for the entire test set.")
            
            # Plot Amp/Freq mismatch vs massarrays
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.plot(massarr, dfmm['mismatch_amp'], 'o', label='Amplitude',
                    markersize=3, alpha=0.5, markeredgewidth=0.25, markeredgecolor='black')
            ax.plot(massarr, dfmm['mismatch_freq'], 's', label='Frequency',
                    markersize=3, alpha=0.5, markeredgewidth=0.25, markeredgecolor='black')
            ax.set_xlabel(xname, fontsize=12)
            ax.set_ylabel('Mismatch', fontsize=12)
            ax.set_yscale('log')  # Set y-axis to logarithmic scale
            # ax.set_title(f'Mismatch vs {xname}', fontsize=12)
            ax.legend(loc='best')
            savename = 'mmplot-alltest-ampfreq-'+xname.replace(' ','')+ '-' + datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.savefig(self.savedir+savename+'.png', dpi=300, bbox_inches='tight', transparent=True)
            logging.debug(f"Mismatch plot saved to {self.savedir+savename}.png")
            plt.close()

            # Plot hplus/hcross mismatch vs massarrays
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.plot(massarr, dfmm['mismatch_hplus'], 'o', label='$h_{+}$',
                    markersize=3, alpha=0.5, markeredgewidth=0.25, markeredgecolor='black')
            ax.plot(massarr, dfmm['mismatch_hcross'], 's', label='$h_{\\times}$',
                    markersize=3, alpha=0.5, markeredgewidth=0.25, markeredgecolor='black')
            ax.set_xlabel(xname, fontsize=12)
            ax.set_ylabel('Mismatch', fontsize=12)
            ax.set_yscale('log')  # Set y-axis to logarithmic scale
            # ax.set_title(f'Mismatch vs {xname}', fontsize=12)
            ax.legend(loc='best')
            savename = 'mmplot-alltest-hphc-'+xname.replace(' ','')+ '-' + datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.savefig(self.savedir+savename+'.png', dpi=300, bbox_inches='tight', transparent=True)
            logging.info(f"Mismatch plot saved to {self.savedir+savename}.png")
            plt.close()
        # Log mean and median mismatch values
        mean_mismatch_amp = np.mean(dfmm['mismatch_amp'])
        median_mismatch_amp = np.median(dfmm['mismatch_amp'])
        mean_mismatch_freq = np.mean(dfmm['mismatch_freq'])
        median_mismatch_freq = np.median(dfmm['mismatch_freq'])
        mean_mismatch_hplus = np.mean(dfmm['mismatch_hplus'])
        median_mismatch_hplus = np.median(dfmm['mismatch_hplus'])
        mean_mismatch_hcross = np.mean(dfmm['mismatch_hcross'])
        median_mismatch_hcross = np.median(dfmm['mismatch_hcross'])
        logging.info(f"Mean Mismatch (Amplitude): {mean_mismatch_amp:.2e}")
        logging.info(f"Median Mismatch (Amplitude): {median_mismatch_amp:.2e}")
        logging.info(f"Mean Mismatch (Frequency): {mean_mismatch_freq:.2e}")
        logging.info(f"Median Mismatch (Frequency): {median_mismatch_freq:.2e}")
        logging.info(f"Mean Mismatch (hplus): {mean_mismatch_hplus:.2e}")
        logging.info(f"Median Mismatch (hplus): {median_mismatch_hplus:.2e}")
        logging.info(f"Mean Mismatch (hcross): {mean_mismatch_hcross:.2e}")
        logging.info(f"Median Mismatch (hcross): {median_mismatch_hcross:.2e}")
        print("All mismatch plots generated for the test set.")


    def test_uq(self):
        """
        Test the uncertainty quantification (UQ) of the model for 1000 random
        sample generation corresponding the same input parameters. The output
        or the error can be visualized as mismatch values for each of these
        generated compared to the actual waveform. Ideally, if our model training
        is perfect, all the mismatches should be the same!

        TODO: Have a dataloader such that it can load specific values from the
        test dataset. Perhaps, need to modify the test dataset classes.
        """
        Nruns = 1000
        logging.info(f"Testing with model: {self.model_path}")
        # Load the trained model
        preset_array_size = 8190 if args.fcutoff else PRESET_ARRAY_SIZE
        model = CVAE(input_shape=(2, preset_array_size), num_classes=2, 
                    key_shape=(2,2)).to(args.device)
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        model.to(device)
        model.eval()
        logging.info("Model loaded and set to evaluation mode.")

        fig, axes = plt.subplots(1, 2, figsize=(10,5))

        # `batch_size`=1, so that we can directly send the full batch for test!
        # Check if the specific value exists in labels
        specific_value = [10, 10]  # Replace with the desired label value
        test_loader = self.setdataloader(batch_size=1)
        for batch in iter(test_loader):
            x, labels, keys, phases, attr = batch
            if any((labels == torch.tensor(specific_value)).all(dim=1)):
                idx = (labels == torch.tensor(specific_value)).all(dim=1).nonzero(as_tuple=True)[0].item()
                x, labels, keys, phases, attr = x[idx], labels[idx], keys[idx], phases[idx], attr
                print(f"Found specific value {specific_value} in the test set.")
                break
        # Move labels to the appropriate device
        labels = labels.to(device)
        logging.info(f'Choosing to test sample {labels}')
        mmtot_amp, mmtot_freq = [], []
        mmtot_hplus, mmtot_hcross = [], []
        for i in range(Nruns):
            with torch.no_grad():
                logging.debug(f'Testing run: {i}')
                # Use the label-conditioned encoders and decoder to generate data
                z1_mean, z1_log_var = model.encode_label_for_x(labels)
                z1p_mean, z1p_log_var = model.encode_label_for_key(labels)
                z1 = model.reparameterize(z1_mean, z1_log_var)
                z1p = model.reparameterize(z1p_mean, z1p_log_var)
                reconst = model.decode(z1, z1p, labels)
                x, reconst, phase = removezeros(x, reconst, phases, attr)
                mismatch_amp, mismatch_freq, chirpmasses, totalmasses, massratios \
                    = plot_mismatch(x, reconst, labels, keys, savedir=self.savedir, nobatchwiseplot=True)
                logging.debug("Amplitude and Frequency mismatch calculated for current batch.")
                logging.debug("Calculating hplus/hcross mismatch for current batch.")
                mismatch_hplus, mismatch_hcross, chirpmasses, totalmasses, massratios \
                    = plot_polarization_mismatch(x, reconst, labels, keys, phases, 
                                                savedir=self.savedir, nobatchwiseplot=True,
                                                num_saved_overplots=None)
                axes[0].plot(i, mismatch_amp.flatten(), '.', color='grey',
                        markersize=4, markeredgewidth=0.25, markeredgecolor='black')
                axes[0].plot(i, mismatch_freq.flatten(), 'x', color='grey',
                        markersize=4, markeredgewidth=0.25, markeredgecolor='black')
                axes[1].plot(i, mismatch_hplus.flatten(), '.', color='grey',
                        markersize=4, markeredgewidth=0.25, markeredgecolor='black')
                axes[1].plot(i, mismatch_hcross.flatten(), 'x', color='grey',
                        markersize=4, markeredgewidth=0.25, markeredgecolor='black')
                mmtot_amp.append(mismatch_amp.flatten()[0])
                mmtot_freq.append(mismatch_freq.flatten()[0])
                mmtot_hplus.append(mismatch_hplus.flatten()[0])
                mmtot_hcross.append(mismatch_hcross.flatten()[0])
        axes[0].legend(['Amplitude', 'Frequency'], loc='upper right')
        axes[1].legend(['$h_{+}$', '$h_{\\times}$'], loc='upper right')
        label = f'$m_1$={labels[0][0]:.2f}, $m_2$={labels[0][1]:.2f}' + ' $M_{\\odot}$'
        for ax in [axes[0], axes[1]]:
            ax.set_yscale('log')
            ax.set_xlabel('Sample', fontsize=12)
            ax.set_ylabel('Mismatch', fontsize=12)
            ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
            ax.yaxis.set_minor_locator(tck.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
        axes[0].text(0.05, 0.025, label, transform=axes[0].transAxes, ha='left', fontsize=12)
        axes[1].text(0.05, 0.95, label, transform=axes[1].transAxes, ha='left', fontsize=12)
        mu_amp, std_amp = np.mean(mmtot_amp), np.std(mmtot_amp)
        mu_freq, std_freq = np.mean(mmtot_freq), np.std(mmtot_freq)
        mu_hplus, std_hplus = np.mean(mmtot_hplus), np.std(mmtot_hplus)
        mu_hcross, std_hcross = np.mean(mmtot_hcross), np.std(mmtot_hcross)
        logging.info(f'Mean Frequency Mismatch: {mu_freq:.2e} ± {std_freq:.2e}')
        logging.info(f'Mean Amplitude Mismatch: {mu_amp:.2e} ± {std_amp:.2e}')
        logging.info(f'Mean hplus Mismatch: {mu_hplus:.2e} ± {std_hplus:.2e}')
        logging.info(f'Mean hcross Mismatch: {mu_hcross:.2e} ± {std_hcross:.2e}')
        axes[0].text(0.05, 0.03,  f'$|\\delta A|$={mu_amp:.2e}' + ', ' +
                    f'$|\\delta f|$={mu_freq:.2e}\n', transform=axes[0].transAxes, ha='left', fontsize=12)
        axes[1].text(0.05, 0.90, '$|\\delta h_{+}|$='+f'{mu_hplus:.2e}' + ', ' +
                    '$|\\delta h_{\\times}|$='+f'{mu_hcross:.2e}', transform=axes[1].transAxes, ha='left', fontsize=12)
        plt.tight_layout()
        figname = f'{self.savedir}/uq-test-' + datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(figname+'.png', dpi=300, transparent=True)
        plt.savefig(figname+'-white.png', dpi=300)
        plt.close()
        print("All UQ tests completed.")

    def test_timecomplexity(self, num=100):
        """
        Test the time complexity of the model for generating a 1-10e4 ish number of samples.
        This is useful for understanding the efficiency of the model in real-time
        applications.
        """
        # Nruns = [1, 10, 50, 100, 500, 1e3, 5e3, 1e4]
        Nruns = np.logspace(0, 5, num=num, dtype=int)
        # Nruns = [int(n) for n in [1, 10, 50, 100, 500, 1e3, 5e3, 1e4, 5e4]]
        logging.info(f"Testing with model: {self.model_path}")
        # Load the trained model
        preset_array_size = 8190 if args.fcutoff else PRESET_ARRAY_SIZE
        model = CVAE(input_shape=(2, preset_array_size), num_classes=2, 
                    key_shape=(2,2)).to(args.device)
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        model.to(device)
        model.eval()
        logging.info("Model loaded and set to evaluation mode.")

        times = []
        for Nr in Nruns:
            # Generate random labels within the training range
            m1 = np.random.uniform(5, 75, Nr)
            m2 = np.random.uniform(5, 75, Nr)
            labels = torch.tensor(np.vstack((m1, m2)).T, dtype=torch.float32).to(device)
            logging.info(f'Choosing to test sample size {labels.shape}')
            import time
            start_time = time.time()
            with torch.no_grad():
                # Use the label-conditioned encoders and decoder to generate data
                z1_mean, z1_log_var = model.encode_label_for_x(labels)
                z1p_mean, z1p_log_var = model.encode_label_for_key(labels)
                z1 = model.reparameterize(z1_mean, z1_log_var)
                z1p = model.reparameterize(z1p_mean, z1p_log_var)
                reconst = model.decode(z1, z1p, labels)
            end_time = time.time()
            elapsed_time = end_time - start_time
            times.append(elapsed_time)
            logging.info(f'Time taken to generate {Nr} samples: {elapsed_time:.4f} seconds')
        # Plot the time complexity results
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(Nruns, times, 'o', color='grey', markersize=6,
                markeredgewidth=0.25, markeredgecolor='black')
        ax.set_xlabel('Number of Samples', fontsize=12)
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.xaxis.set_minor_locator(tck.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
        ax.yaxis.set_minor_locator(tck.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
        ax.text(0.05, 0.95, f'N={len(Nruns)}', transform=ax.transAxes, fontsize=10, verticalalignment='top')
        plt.tight_layout()
        figname = f'{self.savedir}/timecomplexity-test-' + datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(figname+'.png', dpi=300, transparent=True)
        plt.savefig(figname+'-white.png', dpi=300)
        plt.close()



def removezeros(x, reconst, phase, attr):
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
        phase = phase[:, :attr['padded_at']]
        logging.info(f'Removed zero padding from input and reconstructed data.')
    return x, reconst, phase

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

    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
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
    # plt.show()
    plt.close()


def plot_overplot(x, reconst, labels, keys, savename='../results/overplot',
                  reshape2orig=False, transparent=True):
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
    
    fig, axes = plt.subplots(1, 2, figsize=(5, 2))
    
    for j in range(1):
        i = np.random.randint(0, 49, size=1)
        logging.debug(f"Plotting sample {i} with label {labels[i]}")
        if reshape2orig:
            # Reshape to original data shape
            orig_data = x[i].reshape([2,PRESET_ARRAY_SIZE])
            reconst = reconst[i].reshape([2,PRESET_ARRAY_SIZE])
        else:
            # Use the original shape of the data
            logging.debug(f"x shape: {x.shape}, reconst shape: {reconst.shape}")
            orig_data = x[i].reshape([2,x.shape[2]])
            recon_data = reconst[i].reshape([2,reconst.shape[2]])
        
        orig_amp, orig_freq = orig_data[0], orig_data[1]
        recon_amp, recon_freq = recon_data[0], recon_data[1]

        logging.debug(f"Type of labels[{i}]: {type(labels[i])}, Shape: {labels[i].shape}")
        label = labels[i].reshape([2])
        logging.debug(f"Type of reshaped label: {type(label)}, Shape: {label.shape}")
        
        # NOTE: The overplot waveforms are not normalized!
        # Obtain the keys for normalization
        # key = keys[i].reshape([2,2])
        # amp_mean, amp_std = key[0][0], key[0][1]
        # freq_mean, freq_std = key[1][0], key[1][1]

        # # De-normalize the original data!
        # orig_amp = (orig_amp * amp_std) + amp_mean
        # orig_freq = (orig_freq * freq_std) + freq_mean
        # # De-normalize the reconstructed data!
        # recon_amp = (recon_amp * amp_std) + amp_mean
        # recon_freq = (recon_freq * freq_std) + freq_mean
        
        axes[0].plot(np.arange(len(orig_amp)), orig_amp, '-', label=f"Original")
        axes[0].plot(np.arange(len(recon_amp)), recon_amp, '-', label=f"Reconstructed")
        axes[1].plot(np.arange(len(orig_freq)), orig_freq, '-', label=f"Original")
        axes[1].plot(np.arange(len(recon_freq)), recon_freq, '-', label=f"Reconstructed")
        axes[1].set_title(f'$m_1$={float(label[0])}, $m_2$={float(label[1])}', fontsize=8)

    for i, axlabel in enumerate(['Amplitude', 'Frequency']):
        axes[i].set_xlabel('Sample length', fontsize=12)
        axes[i].set_ylabel(axlabel, fontsize=12)
    axes[1].legend(fontsize=8, loc='upper left')
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.2)
    # putils.beautifyPlot(axes)
    if savename:
        savename += '-' + datetime.now().strftime('%Y%m%d_%H%M%S')
        if transparent:
            plt.savefig(savename+'.png', dpi=300, bbox_inches='tight', transparent=True)
        else:
            plt.savefig(savename+'.png', dpi=300, bbox_inches='tight')
        logging.info(f"Overplot saved to {savename}")
    # plt.show()
    plt.close()


def plot_hphc_overplot(hp_orig, hc_orig, hp_recon, hc_recon, label,
                       savename='../results/overplot-hphc', transparent=True):
    """
    Plot the overlaid waveforms for the reconstructed polarization waveforms.

    Parameters:
    -----------
    hp_orig : np.ndarray
        The original h_plus waveform.
    hc_orig : np.ndarray
        The original h_cross waveform.
    hp_recon : np.ndarray
        The reconstructed h_plus waveform.
    hc_recon : np.ndarray
        The reconstructed h_cross waveform.

    Returns:
    --------
    None

    Outputs:
    -------
    Displays a plot of the overlaid waveforms.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 6), width_ratios=[3, 1])
    axes = axes.flatten()    
    for ax in [axes[0], axes[2]]:
        ax.sharex(axes[0])

    axes[0].plot(np.arange(len(hp_orig)), hp_orig, label='Original', color='blue')
    axes[0].plot(np.arange(len(hp_recon)), hp_recon, label='Reconstructed', linestyle='--', color='orange')

    axes[2].plot(np.arange(len(hc_orig)), hc_orig, label='Original', color='blue')
    axes[2].plot(np.arange(len(hc_recon)), hc_recon, label='Reconstructed', linestyle='--', color='orange')
    axes[2].set_title(f'$m_1$={float(label[0])}, $m_2$={float(label[1])}', fontsize=8)

    # Zoomed view near the maximum
    zoom_halfwidth = 75
    zoom_start = max(0, np.argmax(hp_orig) - zoom_halfwidth)
    zoom_end = min(len(hp_orig), np.argmax(hp_orig) + zoom_halfwidth)
    axes[1].plot(np.arange(zoom_start, zoom_end), hp_orig[zoom_start:zoom_end], color='blue')
    axes[1].plot(np.arange(zoom_start, zoom_end), hp_recon[zoom_start:zoom_end], linestyle='--', color='orange')

    zoom_start = max(0, np.argmax(hc_orig) - zoom_halfwidth)
    zoom_end = min(len(hc_orig), np.argmax(hc_orig) + zoom_halfwidth)
    axes[3].plot(np.arange(zoom_start, zoom_end), hc_orig[zoom_start:zoom_end], color='blue')
    axes[3].plot(np.arange(zoom_start, zoom_end), hc_recon[zoom_start:zoom_end], linestyle='--', color='orange')

    # TODO: Maybe I can have time on the x-axis!
    for i in range(0,4):
        axes[i].set_xlabel('Sample length', fontsize=12)
    axes[0].set_ylabel('$h_{+}$', fontsize=12)
    axes[2].set_ylabel('$h_{\\times}$', fontsize=12)
    axes[0].legend(fontsize=8, loc='upper left')
    axes[2].legend(fontsize=8, loc='upper left')
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.2)
    # putils.beautifyPlot(axes)
    if savename:
        savename += '-' + datetime.now().strftime('%Y%m%d_%H%M%S')
        if transparent:
            plt.savefig(savename+'.png', dpi=300, bbox_inches='tight', transparent=True)
        else:
            plt.savefig(savename+'.png', dpi=300, bbox_inches='tight')
        logging.info(f"Overplot saved to {savename}")
    # plt.show()
    plt.close()


def calculate_mismatch(target, reconstructed):
    """
    Calculate the normalized mismatch between the target and reconstructed waveforms.
    Mismatch = 1 - ( <a|b> / sqrt(<a|a> * <b|b>) )
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

def plot_mismatch(x, reconst, labels, keys, reshape2orig=False, savedir='../results/',
                  nobatchwiseplot=False):
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
            logging.debug(f"x shape: {x.shape}, reconst shape: {reconst.shape}")
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

    if not nobatchwiseplot:
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
            logging.debug(f"Mismatch plot saved to {savedir+savename}.png")
            plt.close()
    return mismatch_amp, mismatch_freq, chirpmasses, totalmasses, massratios


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
    from pycbc.filter import match as matchfunc
    from pycbc.psd import aLIGOZeroDetHighPower
    from pycbc.types import TimeSeries

    if isinstance(hp_orig, torch.Tensor):
        hp_orig = hp_orig.detach().cpu().numpy()
    if isinstance(hp_recon, torch.Tensor):
        hp_recon = hp_recon.detach().cpu().numpy()
    
    psd = aLIGOZeroDetHighPower(length=sample_len,
                                delta_f=1.0/(len(hp_orig)*DELTA_T),  # NOT sample_len = duration * sample_rate
                                low_freq_cutoff=f_lower)
    
    logging.debug(f"PSD delta_f: {1.0/(len(hp_orig)*DELTA_T)}")
    # Ensure all arrays are float64 for precision match
    hp_orig = np.asarray(hp_orig, dtype=np.float64)
    hp_recon = np.asarray(hp_recon, dtype=np.float64)
    psd = psd.astype(np.float64)

    hp_orig = TimeSeries(hp_orig, delta_t=DELTA_T)
    hp_recon = TimeSeries(hp_recon, delta_t=DELTA_T)
    logging.debug(f"hp_orig sample rate: {hp_orig.sample_rate}, hp_recon sample rate: {hp_recon.sample_rate}")
    logging.debug(f"hp_orig delta_f: {hp_orig.delta_f}")
    logging.debug(f'len(hp_orig)={len(hp_orig)}, len(hp_recon)={len(hp_recon)}, \
                  len(psd)={len(psd)}')

    match, i = matchfunc(hp_orig, hp_recon, psd=psd, low_frequency_cutoff=f_lower)
    logging.debug(f"Match value: {match}, Index: {i}")
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

def plot_polarization_mismatch(x, reconst, labels, keys, phases, reshape2orig=False,
                               savedir='../results/', nobatchwiseplot=False,
                               num_saved_overplots=0):
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
    phases = phases.cpu().numpy()
    logging.debug(f"x shape: {x.shape}, reconst shape: {reconst.shape}, phases shape: {phases.shape}")

    chirpmasses = np.zeros((labels.shape[0], 1))  # Store chirp masses for each sample
    totalmasses = np.zeros((labels.shape[0], 1))  # Store total masses for each sample
    massratios = np.zeros((labels.shape[0], 1))  # Store mass ratios for each sample
    mismatch_hplus = np.zeros((x.shape[0], 1))
    mismatch_hcross = np.zeros((x.shape[0], 1))

    # iterate over all the samples in one batch
    for i in range(x.shape[0]):
        logging.debug(f"Processing sample {i}")
        if reshape2orig:
            # Reshape to original data shape
            orig_data = x[i].reshape([2,PRESET_ARRAY_SIZE])
            reconst = reconst.reshape([2,PRESET_ARRAY_SIZE])
            phase = phase[i].reshape([1,PRESET_ARRAY_SIZE])
        else:
            # Use the original shape of the data
            orig_data = x[i].reshape([2,x.shape[2]])
            recon_data = reconst[i].reshape([2,reconst.shape[2]])
            logging.debug(f"orig_data shape: {orig_data.shape}, recon_data shape: {recon_data.shape}")
            phase = phases[i].reshape([phases.shape[1]])          
            logging.debug(f'phase shape: {phase.shape}')

        orig_amp, orig_freq = orig_data[0], orig_data[1]
        recon_amp, recon_freq = recon_data[0], recon_data[1]
        
        # Obtain the keys for normalization
        key = keys[i].reshape([2,2])
        amp_mean, amp_std = key[0][0], key[0][1]
        freq_mean, freq_std = key[1][0], key[1][1]

        # De-normalize the original data!
        orig_amp = (orig_amp * amp_std) + amp_mean
        orig_freq = (orig_freq * freq_std) + freq_mean
        logging.debug(f"original amp shape: {orig_amp.shape}, freq shape: {orig_freq.shape}")
        # De-normalize the reconstructed data!
        recon_amp = (recon_amp * amp_std) + amp_mean
        recon_freq = (recon_freq * freq_std) + freq_mean
        logging.debug(f"reconstructed amp shape: {recon_amp.shape}, freq shape: {recon_freq.shape}")

        # # check length of phase array
        # # NOTE: This happens because of the f-cutoff datacase!
        # if len(phase) != orig_amp.shape[0]:
        #     logging.warning(f"Phase shape {phase.shape} does not match input shape {orig_amp.shape}. Adjusting phase.")
        #     # Adjust phase to match orig_amp length
        #     if len(phase) > orig_amp.shape[0]:
        #         phase = phase[1:]
        #     assert len(phase) == orig_amp.shape[0]

        # Combine original Amp/Freq to hplus/hcross
        hp_orig = orig_amp * np.cos(phase)  # this is original phase
        hc_orig = orig_amp * np.sin(phase)
        logging.debug(f"Original hplus shape: {hp_orig.shape}, hcross shape: {hc_orig.shape}")

        # Calculate hplus/hcross for reconstructed data
        hp_recon, hc_recon = polarizations_from_ampfreq(recon_amp, recon_freq)
        if num_saved_overplots is not None:
            if num_saved_overplots <= 10:
                plot_hphc_overplot(hp_orig, hc_orig, hp_recon, hc_recon, label=labels[i],
                                savename=savedir+'overplot-hphc')
                num_saved_overplots = 11

        # Calculate mismatch for hplus and hcross
        mismatch_hplus[i] = calc_polarization_mismatch(hp_orig, hp_recon)
        mismatch_hcross[i] = calc_polarization_mismatch(hc_orig, hc_recon)
        logging.debug(f"Mismatch for hplus: {mismatch_hplus[i]}, hcross: {mismatch_hcross[i]}")

        # Calculate chirp mass
        m1, m2 = labels[i][0], labels[i][1]
        chirp_mass = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
        logging.debug(f"Chirp mass for sample {i}: {chirp_mass}")
        chirpmasses[i] = chirp_mass
        totalmasses[i] = m1 + m2
        massratios[i] = m1 / m2

    if not nobatchwiseplot:
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
    return (mismatch_hplus, mismatch_hcross, chirpmasses, totalmasses, massratios, num_saved_overplots)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Non-Eccentric GW Generator')

    # parser.add_argument('--nsamples', action='store', default=1000, type=int,
    #                         help='default=%(default)s')
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
    parser.add_argument('--fcutoff', action='store_true', default=False,
                            help='use the data which has equal duration samples with \
                                variable frequency cutoff (default=%(default)s)')
    parser.add_argument('--aligned', action='store_true', default=False,
                            help='use the aligned-spin data uniformly sampled in (m1,m2,spin1z,spin2z) space.\
                                (default=%(default)s)')

    parser.add_argument('--test', action='store_true', default=False,
                            help='whether to test?')
    parser.add_argument('--test-uq', action='store_true', default=False,
                        help='whether to test uncertainty quantification?')
    parser.add_argument('--time-complexity', action='store_true', default=False,
                        help='whether to test time complexity?')
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

    # Create results directory for today if it doesn't exist
    today = datetime.today().strftime('%Y%m%d')
    log_dir = f'../results/{today}/'
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    # Set up logging to both console and file
    log_file = os.path.join(log_dir, f'training_{today}.log')
    logging.basicConfig(
        format='%(levelname)s | %(asctime)s: %(message)s',
        level=log_level,
        datefmt='%y-%m-%d %H:%M:%S',
        force=True,
        handlers=[
            logging.StreamHandler(),  # Log to console
            logging.FileHandler(log_file)  # Log to file
        ]
    )
    
    # TODO: Sometimes, `cuda` is not available, and `torch.cuda.is_available()` just goes dead,
    # with no error message. This last happended on 2025-07-22, right during and after a maintanence!
    # 250919: Most likely this is solved now!
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    # device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    args.device = device
    logging.info(f"using {device} device !")

    if args.test:
        try:
            if args.test_uq:
                Test(args).test_uq()
            elif args.time_complexity:
                for n in [100, 500, 1000]:
                    Test(args).test_timecomplexity(n)
            else:
                Test(args).test()
        except RuntimeError as e:
            logging.error(f"Error occurred during testing (perhaps try `--fcutoff`): {e}")
    else:
        # try:
        #     # Always reads data from HDF file now!!
        #     train(args)
        # except RuntimeError as e:
        #     logging.error(f"Error occurred during training (perhaps, ya forgot to put `--fcutoff`): {e}")
        train(args)
