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
import csv
import json
import time
import argparse
import h5py
import numpy as np
import pandas as pd

# import matplotlib
# matplotlib.use('Agg')   # non GUI backend
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib.colors import LogNorm

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

from datacvae import CustomDataset, CustomDataLoader
from datacvae import PRESET_ARRAY_SIZE, SAMPLE_RATE, DELTA_T, f_lower, sample_len
from cvae import CVAE, CAE

from utils import polarizations_from_ampfreq, calc_polarization_mismatch

# from data import SEOBNRv4

import random

markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'X', 'h', '1', '2', '3', '4', '8']

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    PRECISION = 'float64'  # Use double precision for CUDA if available
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    PRECISION = 'float32'  # Use float32 for MPS since it does not support float64 well
else:
    DEVICE = torch.device("cpu")
    PRECISION = 'float64'  # Use double precision for CPU
print(f"Using device: {DEVICE}, with precision: {PRECISION}")

BASE_MODEL_CONFIG = {
    'modeltype': 'cvae',
    'latent_dim_x': 8,
    'latent_dim_key': 3,
    'paramsmean': False,  # whether to use mean and std of labels for normalization
}

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
    os.makedirs(savedir, exist_ok=True)
    os.makedirs('../trained-models', exist_ok=True)

    noklloss = True if args.modeltype=='cae' else False

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
    elif args.aligned:
        num_classes = 4  # m1, m2, spin1z, spin2z
        trainhdf += '-100000-fcutoff-uniform-aligned-regen'
        # trainhdf += '-4e5-fcutoff-uniform-aligned'
        validhdf += '-100000-fcutoff-uniform-aligned-regen'

    if args.dummy:
        # -- use validation set for training, for quick code check!
        trainhdf = args.datadir+args.approximant+'-train-100-fcutoff-uniform-aligned'
        validhdf = args.datadir+args.approximant+'-val-100-fcutoff-uniform-aligned'

    if not os.path.isfile(trainhdf + '.hdf'):
        raise FileNotFoundError(f"Training data file not found: {trainhdf}.hdf")
    if not os.path.isfile(validhdf + '.hdf'):
        raise FileNotFoundError(f"Validation data file not found: {validhdf}.hdf")

    # Try opening the HDF file in read-only mode to check for corruption
    try:
        with h5py.File(trainhdf + '.hdf', 'r') as f:
            logging.info(f"Successfully opened {trainhdf}.hdf in read-only mode.")
    except Exception as e:
        logging.error(f"Error opening {trainhdf}.hdf: {e}")
        raise RuntimeError(f"Could not open {trainhdf}.hdf. The file may be corrupted.")

    logging.info(f'Reading training data from {trainhdf}.hdf')
    train_set = CustomDataset(forwhat='train', approximant=args.approximant, returnattr=True,
                            convert=args.convert, hdf_fname=trainhdf, 
                            train_device=DEVICE, precision=PRECISION)
    logging.info(f'Reading validation data from {validhdf}.hdf')
    valid_set = CustomDataset(forwhat='valid', approximant=args.approximant, returnattr=True,
                            convert=args.convert, hdf_fname=validhdf, 
                            train_device=DEVICE, precision=PRECISION)
    # logging.info(f"Train set size: {len(train_set)}")
    # logging.info(f"Validation set size: {len(valid_set)}")
            
    # try:
    #     training_loader = CustomDataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    #     validation_loader = CustomDataLoader(valid_set, batch_size=args.batch_size, shuffle=True)
    # except ValueError:
    #     training_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    #     validation_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True)

    # -- data loaders return [inputs, target, labels, keys] list!
    training_loader = CustomDataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    validation_loader = CustomDataLoader(valid_set, batch_size=args.batch_size, shuffle=True)
    logging.info(training_loader.__dict__)
    ntbatches = len(training_loader)
    nvbatches = len(validation_loader)
    logging.info(f'Number of Training batches: {ntbatches}')  # this doesn't return the batchsize!
    logging.info(f'Number of Validationg batches: {nvbatches}')

    # -- get mean and std of labels for normalization
    params_fname = '../data/params-' + args.approximant + '-train-100000-fcutoff-uniform-aligned-regen'
    params_df = pd.read_csv(params_fname+'.csv', index_col=0, sep=',')
    params_mean = params_df.mean().values
    params_std = params_df.std().values
    logging.info(f"Labels mean: {params_mean}")
    logging.info(f"Labels std: {params_std}")
    params_mean = torch.tensor(params_mean, dtype=getattr(torch, PRECISION)).to(DEVICE)
    params_std = torch.tensor(params_std, dtype=getattr(torch, PRECISION)).to(DEVICE)

    MODEL_CONFIG = BASE_MODEL_CONFIG.copy()
    if not args.use_base_model_config:
        MODEL_CONFIG['paramsnorm'] = True
        MODEL_CONFIG['labels_mean'] = params_mean
        MODEL_CONFIG['labels_std'] = params_std
        MODEL_CONFIG['num_classes'] = num_classes
        MODEL_CONFIG['latent_dim_x'] = 32
        MODEL_CONFIG['latent_dim_key'] = 2
        MODEL_CONFIG['learning_rate'] = 1e-3
        MODEL_CONFIG['modeltype'] = args.modeltype
    logging.info(f'Model Config: {MODEL_CONFIG}')

    # Initialize Model
    # `num_classes` is the size of the labels.
    # if args.fcutoff or args.aligned:
    PRESET_ARRAY_SIZE = 8191
    if args.modeltype=='cae':
        logging.info(f'Using model type: CAE with num_classes={num_classes} and preset_array_size={PRESET_ARRAY_SIZE}')
        model = CAE(input_shape=(2, PRESET_ARRAY_SIZE), num_classes=num_classes, key_shape=(2,2),
                    MODEL_CONFIG=MODEL_CONFIG)
    else:
        logging.info(f'Using model type: CVAE with num_classes={num_classes} and preset_array_size={PRESET_ARRAY_SIZE}')
        model = CVAE(input_shape=(2, PRESET_ARRAY_SIZE), num_classes=num_classes, key_shape=(2,2),
                    MODEL_CONFIG=MODEL_CONFIG)

    if args.model is not None:
        model_path = '../trained-models/' + args.model
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        logging.info(f"Loaded model from {model_path}")

    # -- move model to device and convert to double precision
    model.to(DEVICE)
    model.to(getattr(torch, PRECISION))

    # Add a learning rate scheduler
    # Scheduler will adjust learning rate after every epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=MODEL_CONFIG.get('learning_rate', 1e-4))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #             optimizer, 
    #             mode='min', 
    #             factor=0.5, 
    #             patience=2, 
    #             threshold=1e-5)
    logging.info('Model Initialized')

    logging.info(f'Starting Training with: {args}')
    train_rloss, valid_rloss = [], []  # running loss every batch
    train_loss, valid_loss = [], [] 
    netreconloss, netklloss, netmmloss = [], [], []
    netvreconloss, netvklloss, netvmmloss = [], [], []
    rloss_train_eval, rloss_recon_eval, rloss_kl_eval = [], [], []
    for epoch in tqdm(range(args.epochs), desc='Epoch'):
        model.train(True)
        # avg_loss = train_one_epoch(training_loader, epoch)

        # -- Since, I want to rewrite the HDF file in the first epoch,
        # -- after that is done, changes are only visible to the next
        # -- epoch if the HDF file is properly closed after writing. 
        # -- So, for epoch==1, I will first close the training_loader and
        # -- validation_loader and then reopen them, so that the changes are 
        # -- visible to the next epoch.
        if epoch==1 and 'regen' not in trainhdf:
            del training_loader
            del validation_loader
            torch.cuda.empty_cache()  # Clear GPU memory cache to free up memory
            training_loader = CustomDataLoader(train_set, batch_size=args.batch_size, shuffle=True)
            validation_loader = CustomDataLoader(valid_set, batch_size=args.batch_size, shuffle=True)

        # Train model for one Epoch
        for x, target, labels, keys, strains, attr in tqdm(training_loader, total=len(training_loader),
                                            desc='Steps/Batchs'):
            """
            `x` is [freq, amp], `labels` is [m1,m2] etc. and
            `keys` is [[amp-mean,amp-var],[freq-mean,freq-var]]
            """
            x, target, labels, keys = x.to(DEVICE), target.to(DEVICE), labels.to(DEVICE), keys.to(DEVICE)
            strains = strains.to(DEVICE)
            optimizer.zero_grad()
            x_recon, zvars = model(x, labels, keys)
            
            # TODO: have it such that the training target are unnormalized waveforms!
            # loss, reconloss, klloss = model.loss_function(x, x_recon, zvars)
            if noklloss:
                loss, reconloss, mmloss = model.mismatch_nokl_loss_func(target, x_recon, zvars, strains=strains, keys=keys, attr=attr)
            elif args.usemmloss:
                loss, reconloss, klloss, mmloss = model.mismatch_loss_func(target, x_recon, zvars, strains=strains, keys=keys, attr=attr)
            else:
                loss, reconloss, klloss = model.loss_function(target, x_recon, zvars)
            loss.backward()
            # -- perform optimization per batch/step
            optimizer.step()

            train_rloss.append(loss.item())
            netreconloss.append(reconloss.item())
            if not noklloss:
                netklloss.append(klloss.item())
            if args.usemmloss or noklloss:
                netmmloss.append(mmloss.item())

        # -- update learning rate per epoch
        scheduler.step(loss)
        logging.info(f"Epoch {epoch+1} completed. Learning rate adjusted to: {scheduler.get_last_lr()[0]:.2e}")
        logging.debug(f"x shape: {x.shape}, labels shape: {labels.shape}, keys shape: {keys.shape}")

        train_loss.append(loss.item())
        logging.info(f'Average training loss for epoch {epoch+1}: {loss.item():.4f}')

        # NOTE: Validation is performed after all training batches are done
        # per epoch, so the validation loss can be lower than the training loss 
        # for the last batch, since the model has already been updated by the 
        # last training batch before validation. Training loss at the start of
        # each epoch will be larger than the validation loss at the end of the 
        # previous epoch, since the model is updated after validation and before 
        # the next epoch starts. Thus, we have put the optimizer step after
        # validation step, so that the training loss and validation loss are more
        # comparable to each other.

        # -- set model to eval mode for validation
        model.eval()

        # -- Do one cycle training in eval mode with no grad after training is finished,
        # -- to compare train and validation losses at the same epoch and check for overfitting etc.
        train_eval_loss = 0.0
        with torch.no_grad():
            for idx, (x, target, labels, keys, strains, attr) in enumerate(tqdm(training_loader, ncols=80, desc="Train-eval-steps")):
                x, target, labels, keys = x.to(DEVICE), target.to(DEVICE), labels.to(DEVICE), keys.to(DEVICE)
                x_recon, zvars = model(x, labels, keys)
                loss, recon_loss, kl_loss = model.loss_function(target, x_recon, zvars)
                train_eval_loss += loss.item()
                rloss_train_eval.append(loss.item())
                rloss_recon_eval.append(recon_loss.item())
                rloss_kl_eval.append(kl_loss.item())
        avg_train_eval_loss = train_eval_loss / (len(training_loader))
        logging.info(f"Epoch {epoch+1}, Batch Avg Train Eval Loss: {avg_train_eval_loss:.4f}")

        # Evaluate on validation set
        with torch.no_grad():  # Disable gradient computation for validation
            # just reconstruct target and calculate diff
            for vx, target, vlabels, vkeys, vstrains, vattr in tqdm(validation_loader, desc='val-batch'):
                vx, target, vlabels, vkeys = vx.to(DEVICE), target.to(DEVICE), vlabels.to(DEVICE), vkeys.to(DEVICE)
                vx_recon, vzvars = model(vx, vlabels, vkeys)
                if noklloss:
                    vloss, vreconloss, vmmloss = model.mismatch_nokl_loss_func(target, vx_recon, vzvars, strains=vstrains.to(DEVICE), keys=vkeys.to(DEVICE), attr=vattr)
                elif args.usemmloss:
                    vloss, vreconloss, vklloss, vmmloss = model.mismatch_loss_func(target, vx_recon, vzvars, strains=vstrains.to(DEVICE), keys=vkeys.to(DEVICE), attr=vattr)
                else:
                    vloss, vreconloss, vklloss = model.loss_function(target, vx_recon, vzvars)
                valid_rloss.append(vloss.item())
                netvreconloss.append(vreconloss.item())
                if not noklloss:
                    netvklloss.append(vklloss.item())
                if args.usemmloss or noklloss:
                    netvmmloss.append(vmmloss.item())
            valid_loss.append(vloss.item())
        tqdm.write(f'Epoch {epoch+1} : train loss {loss.item()} & valid loss {vloss.item()}')
        
        # Save model checkpoint at every epoch as backup
        backup_model_path = f'../trained-models/model-backup-epoch{epoch}-{timestamp}'
        torch.save(model.state_dict(), backup_model_path)
        logging.info(f"Model backup saved at {backup_model_path}")
    
    savename = timestamp + '-' + str(args.epochs)
    modeldir = f'../trained-models/{datetime.now().strftime("%Y%m%d")}'
    if not os.path.isdir(modeldir):
        os.makedirs(modeldir)
    model_path = f'{modeldir}/model-'
    model_path += 'mmloss-' if args.usemmloss else ''
    model_path += args.modeltype + '-nokll-' if noklloss else ''
    model_path += savename

    if not args.nosave:
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        torch.save(model.state_dict(), model_path)
        # Save losses to files
        # When using pandas, all arrays should have same length
        assert len(train_loss) == len(valid_loss) == args.epochs, "train_loss and valid_loss should have length equal to number of epochs"
        assert len(train_rloss) == ntbatches * args.epochs, "train_rloss should have length equal to number of training batches times number of epochs"
        assert len(valid_rloss) == nvbatches * args.epochs, "valid_rloss should have length equal to number of validation batches times number of epochs"
        assert len(netreconloss) == ntbatches * args.epochs, "netreconloss should have length equal to number of training batches times number of epochs"
        assert len(netvreconloss) == nvbatches * args.epochs, "netvreconloss should have length equal to number of validation batches times number of epochs"
        assert len(rloss_train_eval) == ntbatches * args.epochs, "rloss_train_eval should have length equal to number of training batches times number of epochs"
        assert len(rloss_recon_eval) == ntbatches * args.epochs, "rloss_recon_eval should have length equal to number of training batches times number of epochs"
        assert len(rloss_kl_eval) == ntbatches * args.epochs, "rloss_kl_eval should have length equal to number of training batches times number of epochs"
        if not noklloss:
            assert len(netklloss) == ntbatches * args.epochs, "netklloss should have length equal to number of training batches times number of epochs"
            assert len(netvklloss) == nvbatches * args.epochs, "netvklloss should have length equal to number of validation batches times number of epochs"
        if args.usemmloss or noklloss:
            assert len(netmmloss) == ntbatches * args.epochs, "netmmloss should have length equal to number of training batches times number of epochs"
            assert len(netvmmloss) == nvbatches * args.epochs, "netvmmloss should have length equal to number of validation batches times number of epochs"
        logging.info(f"Lengths of loss arrays are consistent with number of epochs and batches.")
        logging.debug(f"train_loss length: {len(train_loss)}, valid_loss length: {len(valid_loss)}")
        logging.debug(f"train_rloss length: {len(train_rloss)}, valid_rloss length: {len(valid_rloss)}")
        logging.debug(f"netreconloss length: {len(netreconloss)}, netvreconloss length: {len(netvreconloss)}")
        logging.debug(f"rloss_train_eval length: {len(rloss_train_eval)}, rloss_recon_eval length: {len(rloss_recon_eval)}, rloss_kl_eval length: {len(rloss_kl_eval)}")
        if not noklloss:
            logging.debug(f"netklloss length: {len(netklloss)}, netvklloss length: {len(netvklloss)}")
        if args.usemmloss or noklloss:
            logging.debug(f"netmmloss length: {len(netmmloss)}, netvmmloss length: {len(netvmmloss)}")
        dfepoch = pd.DataFrame({
            'train_loss': train_loss,
            'valid_loss': valid_loss,})
        dfepoch.to_csv(savedir + f'epoch-loss-{timestamp}.csv', index=False)
        np.savetxt(savedir + f'train-rloss-{timestamp}.txt', train_rloss)
        np.savetxt(savedir + f'valid-rloss-{timestamp}.txt', valid_rloss)
        dftrain = pd.DataFrame({
            'netreconloss': netreconloss,
            'netklloss': netklloss if not noklloss else [0]*len(netreconloss),
            'netmmloss': netmmloss if args.usemmloss or noklloss else [0]*len(netreconloss),
            })
        dfval = pd.DataFrame({
            'netvreconloss': netvreconloss,
            'netvklloss': netvklloss if not noklloss else [0]*len(netvreconloss),
            'netvmmloss': netvmmloss if args.usemmloss or noklloss else [0]*len(netvklloss),
            })
        dfeval = pd.DataFrame({
            'train_eval_loss': rloss_train_eval,
            'train_eval_recon_loss': rloss_recon_eval,
            'train_eval_kl_loss': rloss_kl_eval,
            })
        savename = args.modeltype + '-nokll-' if noklloss else ''
        dftrain.to_csv(savedir + f'net-train-loss-{savename}{timestamp}.csv', index=False)
        dfval.to_csv(savedir + f'net-val-loss-{savename}{timestamp}.csv', index=False)
        dfeval.to_csv(savedir + f'train-eval-loss-{savename}{timestamp}.csv', index=False)

        # -- Save MODEL_CONFIG to JSON file with same savename for future reference
        model_config_path = f'../trained-models/model-config-{savename}{timestamp}.json'
        # -- Convert any non-serializable objects in MODEL_CONFIG to strings for JSON serialization
        for key, value in MODEL_CONFIG.items():
            if isinstance(value, torch.Tensor):
                MODEL_CONFIG[key] = value.cpu().numpy().tolist()  # Convert tensor to list
            elif isinstance(value, torch.dtype):
                MODEL_CONFIG[key] = str(value)  # Convert dtype to string
        with open(model_config_path, 'w') as f:
            json.dump(MODEL_CONFIG, f)
        logging.info(f"Model config saved at {model_config_path}")

    fig, axes = plt.subplots(2, 1, figsize=(5, 10))
    axes[0].plot(np.arange(args.epochs), train_loss, label='training loss')
    axes[0].plot(np.arange(args.epochs), valid_loss, label='validation loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend()
    axes[1].plot(np.arange(args.epochs*ntbatches), train_rloss, label='train running loss')
    axes[1].plot(np.arange(args.epochs*nvbatches), valid_rloss, label='valid running loss')
    axes[1].plot(np.arange(args.epochs*ntbatches), netreconloss, label='reconstruction loss')
    if not noklloss:
        axes[1].plot(np.arange(args.epochs*ntbatches), netklloss, label='latent loss')
    if args.usemmloss or noklloss:
        axes[1].plot(np.arange(args.epochs*ntbatches), netmmloss, label='mismatch loss')
    axes[1].plot(np.arange(args.epochs*nvbatches), netvreconloss, label='valid reconstruction loss')
    if not noklloss:
        axes[1].plot(np.arange(args.epochs*nvbatches), netvklloss, label='valid latent loss')
    if args.usemmloss or noklloss:
        axes[1].plot(np.arange(args.epochs*nvbatches), netvmmloss, label='valid mismatch loss')
    axes[1].set_xlabel('Batch', fontsize=12)
    axes[1].set_yscale('log')  # Set y-axis to logarithmic scale
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend()
    # putils.beautifyPlot(axes)
    plt.tight_layout()
    if not args.nosave:
        plt.savefig(savedir + f'epoch-loss-{savename}{timestamp}.png', dpi=300)

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
        self.aligned = args.aligned
        self.datadir = args.datadir
        self.testhdf = self.datadir+self.approximant+'-test'
        if args.fcutoff:
            self.testhdf += '-f_cutoff'
        self.batch_size = args.batch_size
        self.modeltype = args.modeltype

        if args.fix_random_seed:
            random_seed = args.random_seed  # -- TODO: make this configurable!
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logging.info(f"Random seed fixed to {random_seed} for reproducibility.")

        # NOTE: Using str values for precision, so that attr are callable for both torch and numpy
        # self.device = args.device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.precision = 'float64'  # Use double precision for CUDA if available
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.precision = 'float32'  # Use float32 for MPS since it does not support float64 well
        else:
            self.device = torch.device("cpu")
            self.precision = 'float64'  # Use double precision for CPU
        logging.info(f"Using device: {self.device}, with precision: {self.precision}")

        if not args.time_complexity and not args.time_compare:
            self.test_loader = self.setdataloader()

        self.noshow = args.noshow
        self.nosave = args.nosave
        self.maxiters = args.nsamples

        today = datetime.today().strftime('%Y%m%d') if args.today is None else args.today
        savedir = args.savedir if args.savedir is not None else '../results'
        if not os.path.isdir(savedir+f'/{today}/'):
            os.makedirs(savedir+f'/{today}/')
        self.savedir = savedir+f'/{today}/'

        self.epochs = 1
        if args.model is not None:
            self.model_path = '../trained-models/' + args.model        
            logging.info('Test DataLoader set up.')

    def _load_model(self):
        """
        Load the trained model from the specified path.
        Depending on the model type (CAE or CVAE), initialize the appropriate 
        model architecture and load the state dictionary.
        Also, gets the mean and std of labels for normalization, 
        which are needed to initialize the model.
        """
        # Load the trained model
        preset_array_size = 8190 if args.fcutoff else PRESET_ARRAY_SIZE
        num_classes = 4 if self.aligned else 2

        # -- get mean and std of labels for normalization
        params_fname = '../data/params-' + args.approximant + '-train-100000-fcutoff-uniform-aligned-regen'
        params_df = pd.read_csv(params_fname+'.csv', index_col=0, sep=',')
        params_mean = params_df.mean().values
        params_std = params_df.std().values
        logging.info(f"Labels mean: {params_mean}")
        logging.info(f"Labels std: {params_std}")
        params_mean = torch.tensor(params_mean, dtype=getattr(torch, self.precision)).to(self.device)
        params_std = torch.tensor(params_std, dtype=getattr(torch, self.precision)).to(self.device)

        MODEL_CONFIG['paramsmean'] = True
        MODEL_CONFIG['labels_mean'] = params_mean
        MODEL_CONFIG['labels_std'] = params_std
        MODEL_CONFIG['num_classes'] = num_classes
        MODEL_CONFIG['latent_dim_x'] = 32
        MODEL_CONFIG['latent_dim_key'] = 2

        if self.modeltype=='cae':
            logging.info(f'Using model type: CAE with num_classes={num_classes} and preset_array_size={preset_array_size}')
            model = CAE(input_shape=(2, preset_array_size), num_classes=num_classes, key_shape=(2,2),
                        MODEL_CONFIG=MODEL_CONFIG)
        else:
            logging.info(f'Using model type: CVAE with num_classes={num_classes} and preset_array_size={preset_array_size}')
            model = CVAE(input_shape=(2, preset_array_size), num_classes=num_classes, key_shape=(2,2),
                        MODEL_CONFIG=MODEL_CONFIG)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        model.to(getattr(torch, self.precision))
        model.eval()  # Set model to evaluation mode
        logging.info(f"Loaded model from {self.model_path}")
        return model


    def load_flex_model(self, configpath, model_path):
        """
        Load the trained FlexTwoC2E1D model from the specified path.
        """
        from flexcvae import FlexTwoC2E1D, FlexCAE, FlexCAEPhase

        logging.info(f"Loading FlexTwoC2E1D model with config from: {configpath} and model weights from: {model_path}")
        if configpath is not None:
            configpath = '../trained-models/' + configpath
            if not configpath.endswith('.json'):
                configpath += '.json'
            if not os.path.isfile(configpath):
                logging.error(f"Provided MODEL_CONFIG path does not exist: {configpath}")
                raise FileNotFoundError(f"MODEL_CONFIG file not found at {configpath}")
            logging.info(f"Using MODEL_CONFIG: {configpath}")
            MODEL_CONFIG = json.load(open(configpath, 'r'))
        else:
            logging.info("No MODEL_CONFIG provided. Using default hyperparameter values.")
            MODEL_CONFIG = BASE_MODEL_CONFIG.copy()

        # -- Load labels mean and std values if provided in MODEL_CONFIG
        if isinstance(MODEL_CONFIG['labels_mean'], str) and isinstance(MODEL_CONFIG['labels_std'], str):
            # print(MODEL_CONFIG['labels_std'])
            # print(MODEL_CONFIG['labels_std'].strip('[]').split(','))
            # print(float(MODEL_CONFIG['labels_std'].strip('[]').split(',')[0]))
            # print(type(MODEL_CONFIG['labels_std'].strip('[]').split(',')[0]))
            logging.info("Converting labels_mean and labels_std from str->lists to numpy arrays for model initialization.")
            if MODEL_CONFIG['labels_mean'] == "None" or MODEL_CONFIG['labels_std'] == "None":
                logging.warning("Labels mean or std is None in MODEL_CONFIG, skipping conversion and normalization.")
                MODEL_CONFIG['labels_mean'] = None
                MODEL_CONFIG['labels_std'] = None
            else:
                MODEL_CONFIG['labels_mean'] = np.array(MODEL_CONFIG['labels_mean'].strip('[]').split(',')).astype(float)
                MODEL_CONFIG['labels_std'] = np.array(MODEL_CONFIG['labels_std'].strip('[]').split(',')).astype(float)
        if MODEL_CONFIG['labels_mean'] is not None and MODEL_CONFIG['labels_std'] is not None:
            # logging.warning('For now we will use predefined global params_mean and params_std for normalization instead of converting from MODEL_CONFIG, since the conversion is not working well and giving NaN values for some reason. This needs to be fixed later.')
            # labels_mean = params_mean.cpu().numpy()
            # labels_std = params_std.cpu().numpy()
            MODEL_CONFIG['labels_mean'] = torch.tensor(MODEL_CONFIG['labels_mean'], dtype=getattr(torch, PRECISION)).to(DEVICE)
            MODEL_CONFIG['labels_std'] = torch.tensor(MODEL_CONFIG['labels_std'], dtype=getattr(torch, PRECISION)).to(DEVICE)

        # Convert some hyperparameters from str to appropriate types if needed (e.g. lists, tuples)
        if isinstance(MODEL_CONFIG['input_shape'], str):
            logging.info("Converting input_shape from str to tuple for model initialization.")
            MODEL_CONFIG['input_shape'] = tuple(map(int, MODEL_CONFIG['input_shape'].strip('()').split(',')))
        if isinstance(MODEL_CONFIG['key_shape'], str):
            logging.info("Converting key_shape from str to tuple for model initialization.")
            MODEL_CONFIG['key_shape'] = tuple(map(int, MODEL_CONFIG['key_shape'].strip('()').split(',')))
        if isinstance(MODEL_CONFIG['target'], str) and MODEL_CONFIG['target'].lower() == 'none':
            logging.info("Setting target to None for model initialization.")
            MODEL_CONFIG['target'] = None

        for k in MODEL_CONFIG:
            if MODEL_CONFIG[k] == "None":
                logging.info(f"Setting {k} to None for model initialization.")
                MODEL_CONFIG[k] = None
            if MODEL_CONFIG[k] == "False":
                logging.info(f"Setting {k} to False for model initialization.")
                MODEL_CONFIG[k] = False
            if MODEL_CONFIG[k] == "True":
                logging.info(f"Setting {k} to True for model initialization.")
                MODEL_CONFIG[k] = True

        if 'modeltype' not in MODEL_CONFIG:
            logging.warning("modeltype not specified in MODEL_CONFIG, defaulting to 'flexcvae'.")
            MODEL_CONFIG['modeltype'] = 'flexcvae'
        if MODEL_CONFIG['modeltype'].lower() not in ['flexcvae', 'flexcae', 'flexcaephase', 'original']:
            logging.error(f"Invalid modeltype specified in MODEL_CONFIG: {MODEL_CONFIG['modeltype']}. Must be 'flexcvae', 'flexcae', or 'flexcaephase'.")
            raise ValueError(f"Invalid modeltype specified in MODEL_CONFIG: {MODEL_CONFIG['modeltype']}. Must be 'flexcvae', 'flexcae', or 'flexcaephase'.")

        if MODEL_CONFIG.get('target', None) is not None:
            logging.info(f"Model will be initialized with target: {MODEL_CONFIG['target']}")

        if MODEL_CONFIG['target'] == 'amp_phase':
            logging.info("Initializing FlexCAEPhase model since target is 'amp_phase'.")
            model = FlexCAEPhase(
                MODEL_CONFIG=MODEL_CONFIG,
                input_shape=(2, PRESET_ARRAY_SIZE),
                num_classes=4,
            )
                # -- For amp-phase target, we need to set the loss function type to 'mismatch_nokl' since KL loss does not make sense for deterministic CAE.
            MODEL_CONFIG['loss_func_type'] = 'mismatch_nokl'
            logging.warning("For 'amp_phase' target, setting loss_func_type to 'mismatch_nokl' since KL loss does not make sense for deterministic CAE.")  
        elif MODEL_CONFIG['modeltype'].lower() == 'flexcae':
            model = FlexCAE(
                MODEL_CONFIG=MODEL_CONFIG,
                input_shape=(2, PRESET_ARRAY_SIZE),
                num_classes=4,
            )
        elif MODEL_CONFIG['modeltype'].lower() == 'original':
            logging.info("Initializing original CVAE model architecture for testing.")
            model = CVAE(input_shape=(2, 8190), 
                         num_classes=4, key_shape=(2,2),
                        MODEL_CONFIG=MODEL_CONFIG)
        else:
            model = FlexTwoC2E1D(
                MODEL_CONFIG=MODEL_CONFIG,
                input_shape=(2, PRESET_ARRAY_SIZE),
                num_classes=4,
            )
        logging.info(f"Model architecture of type {model.__class__.__name__} initialized. Now loading model weights.")
        logging.info(f"Model initialized with the following hyperparameters: {MODEL_CONFIG}")
        # print(model)
        print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"Total number of trainable parameters: \
            {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
        if not os.path.isfile(model_path):
            logging.error(f"Provided model path does not exist: {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
        # -- Check if model weights loaded are of the same precision as our initialized model.
        # -- If not, then convert loaded model to the correct precision before moving to device.
        for name, param in model.named_parameters():
            if param.dtype != getattr(torch, self.precision):
                logging.info(f"Converting model parameter '{name}' from {param.dtype} to {self.precision}")
                param.data = param.data.to(getattr(torch, self.precision))
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        logging.info(f"Loaded model from {model_path}")
        model.to(self.device)
        model.to(getattr(torch, self.precision))
        model.eval()
        logging.info("Model loaded, moved to device, converted to appropriate precision, and set to evaluation mode.")
        return model

    def setdataloader(self, batch_size=None, custom_batch=None):
        """
        Set up the DataLoader for the test dataset.
        """
        batch_size = batch_size if batch_size is not None else self.batch_size
        testhdf = self.testhdf + '-100000-fcutoff-uniform-aligned-regen' if self.aligned else self.testhdf
        if not os.path.isfile('../data/' + testhdf + '.hdf'):
            raise FileNotFoundError(f"Test data file not found: {testhdf}.hdf")
        test_set = CustomDataset(forwhat='test', approximant=self.approximant,
                                convert=self.convert, hdf_fname=testhdf,
                                returnattr=True, train_device=args.device, 
                                precision=self.precision,
                                custom_batch=custom_batch)
        logging.info(f'Reading test data from {testhdf}')
        test_loader = CustomDataLoader(test_set, batch_size=batch_size, shuffle=True)
        logging.info(f'Test set size: {len(test_set)}')
        return test_loader

    def test(self):
        """
        Test the trained CVAE model using only labels as input.
        TODO: Create a test dir in results in dir and a subfolder with timestamp !!
        """
        if not hasattr(self, 'model_path'):
            self.model_path = '../trained-models/model-flexcvae-20260323-165944.pt'
        logging.info(f"Testing with model: {self.model_path}")

        if self.modeltype == 'flexcvae':
            logging.info("Loading FlexTwoC2E1D model for testing.")
            model = self.load_flex_model(configpath='modelconfig-flexcvae-20260323-165944.json', 
                                         model_path=self.model_path)
        else:
            model = self._load_model()
        logging.info("Model loaded and set to evaluation mode.")

        # Initialize dataframe to store mismatch results of whole test set!
        dfmm = pd.DataFrame(columns=['chirp_mass', 'total_mass', 'mass_ratio',
                                     'mismatch_amp', 'mismatch_freq', 
                                     'mismatch_hplus', 'mismatch_hcross',])

        # for _ in range(self.epochs):
            # `next(iter(self.test_loader))` gives us a batch of data!
            # Thus, `shape(x)` is (batch_size, 2, PRESET_ARRAY_SIZE) etc.

        # Iterate over all the batches
        num_saved_overplots, iters = 0, 0
        for (x, labels, keys, phases, strains, attr) in tqdm(iter(self.test_loader)):
            # logging.debug(f"Attributes: {attr}")  # Ensure 'attr' is defined or replace with the correct variable

            # plt.plot(range(len(x[0][0])), x[0][0].cpu().numpy(), label='input')
            # if not self.noshow:
            #     plt.show()
            if self.maxiters is not None and iters >= self.maxiters:
                break
            iters += 1
            
            # Move labels to the appropriate device
            labels = labels.to(self.device)

            # Generate reconstructed data
            with torch.no_grad():
                # Use the label-conditioned encoders and decoder to generate data
                z1_mean, z1_log_var = model.encode_label_for_x(labels)
                z1p_mean, z1p_log_var = model.encode_label_for_key(labels)
                if self.modeltype == 'flexcvae':
                    logging.info("Using FlexTwoC2E1D model for reconstruction.")
                    # z1 = model.reparameterize(z1_mean, z1_log_var)
                    # z1p = model.reparameterize(z1p_mean, z1p_log_var)
                    z = torch.cat((z1_mean, z1p_mean), dim=1)  # Concatenate latent vectors
                    reconst = model.decode(z, labels)
                elif self.modeltype=='cae':
                    reconst = model.decode(z1_mean, z1p_mean, labels)
                else:
                    z1 = model.reparameterize(z1_mean, z1_log_var)
                    z1p = model.reparameterize(z1p_mean, z1p_log_var)
                    reconst = model.decode(z1, z1p, labels)
            logging.debug(f"x shape: {x.shape}, reconst shape: {reconst.shape}, keys shape: {keys.shape}")
            if not self.aligned:
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
            mismatch_hplus, mismatch_hcross, chirpmasses, totalmasses, massratios, chieffs, num_saved_overplots \
                = plot_polarization_mismatch(x, reconst, labels, keys, phases, strains, attr,
                                             savedir=self.savedir, nobatchwiseplot=True,
                                             num_saved_overplots=num_saved_overplots)
            logging.info("hplus/hcross mismatch calculated for current batch.")
            logging.info(f"Tests completed for current batch.")
            # Save the mismatch results to the dataframe
            dfmm = pd.concat([dfmm, pd.DataFrame({
                'mass1': labels[:,0].cpu().numpy(),
                'mass2': labels[:,1].cpu().numpy(),
                'spin1z': labels[:,2].cpu().numpy() if self.aligned else np.zeros(len(labels)),
                'spin2z': labels[:,3].cpu().numpy() if self.aligned else np.zeros(len(labels)),
                'chirp_mass': chirpmasses.flatten(),
                'total_mass': totalmasses.flatten(),
                'mass_ratio': massratios.flatten(),
                'chi_eff': chieffs.flatten(), # if not aligned, then vals are zeros.
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
            ax.tick_params(which='both', direction='in', top=True, right=True)
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
            ax.tick_params(which='both', direction='in', top=True, right=True)
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
        logging.info(f"Best mismatch (hplus): {np.min(dfmm['mismatch_hplus']):.2e}")
        logging.info(f"Best mismatch (hcross): {np.min(dfmm['mismatch_hcross']):.2e}")
        logging.info(f"Worst mismatch (hplus): {np.max(dfmm['mismatch_hplus']):.2e}")
        logging.info(f"Worst mismatch (hcross): {np.max(dfmm['mismatch_hcross']):.2e}")
        print("All mismatch plots generated for the test set.")
        dfmm.to_hdf(self.savedir + 'mismatch-results-' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.h5', key='dfmm', mode='w')

        # Plot mismatchs in the mass ratio and chi_eff plane
        if self.aligned:
            self.plot_mmcontour_in_qchi_space(dfmm)
            self.plot_mm_vs_chieff(dfmm)
            self.plot_mm_vs_spin(dfmm)

    def plot_mmcontour_in_qchi_space(self, dfmm):
        """
        Plot the mismatches in the mass ratio and chi_eff plane as contours.
        """
        logging.info("Plotting mismatches in the mass ratio and chi_eff plane.")
        from scipy.interpolate import griddata

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

        # Plotting
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        contour_levels = np.logspace(-6, 0, 13)
        cs1 = axes[0, 0].contourf(xi, yi, zi_amp, levels=contour_levels, norm=LogNorm(), cmap='viridis')
        fig.colorbar(cs1, ax=axes[0, 0], label='Mismatch Amplitude')
        axes[0, 0].set_title('Mismatch Amplitude')
        axes[0, 0].set_xlabel('Mass Ratio (q)')
        axes[0, 0].set_ylabel('Effective Spin ($\\chi_{\\rm eff}$)')
        cs2 = axes[0, 1].contourf(xi, yi, zi_freq, levels=contour_levels, norm=LogNorm(), cmap='viridis')
        fig.colorbar(cs2, ax=axes[0, 1], label='Mismatch Frequency')
        axes[0, 1].set_title('Mismatch Frequency')
        axes[0, 1].set_xlabel('Mass Ratio (q)')
        axes[0, 1].set_ylabel('Effective Spin ($\\chi_{\\rm eff}$)')
        cs3 = axes[1, 0].contourf(xi, yi, zi_hplus, levels=contour_levels, norm=LogNorm(), cmap='viridis')
        fig.colorbar(cs3, ax=axes[1, 0], label='$h_{+}$ mismatch')
        # axes[1, 0].set_title('Mismatch hplus')
        axes[1, 0].set_xlabel('Mass Ratio (q)')
        axes[1, 0].set_ylabel('Effective Spin ($\\chi_{\\rm eff}$)')
        cs4 = axes[1, 1].contourf(xi, yi, zi_hcross, levels=contour_levels, norm=LogNorm(), cmap='viridis')
        fig.colorbar(cs4, ax=axes[1, 1], label='$h_{\\times}$ mismatch')
        # axes[1, 1].set_title('Mismatch hcross')
        axes[1, 1].set_xlabel('Mass Ratio (q)', fontsize=12)
        axes[1, 1].set_ylabel('Effective Spin ($\\chi_{\\rm eff}$)')
        for ax in axes.flatten():
            ax.tick_params(which='both', direction='in', top=True, right=True)
        plt.tight_layout()
        savename = self.savedir + 'mmcontour_in_qchi_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.png'
        plt.savefig(savename, dpi=300, bbox_inches='tight', transparent=True)
        logging.info(f"Mismatch in q-chi_eff plane plot saved to {savename}")
        plt.close()

    def plot_mm_vs_chieff(self, dfmm):
        """
        Plot the mismatches vs chi_eff.
        """
        logging.info("Plotting mismatches vs chi_eff.")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for i, (mm_col, ax) in enumerate(zip(['mismatch_amp', 'mismatch_freq'], axes)):
            ax.plot(dfmm['chi_eff'], dfmm[mm_col], 'o',
                    markersize=3, alpha=0.5, markeredgewidth=0.25, markeredgecolor='black')
            ax.set_xlabel('Chi_eff', fontsize=12)
            ax.set_ylabel(f'{mm_col}', fontsize=12)
            ax.set_yscale('log')
            ax.set_title(f'{mm_col} vs Chi_eff', fontsize=12)
        plt.tight_layout()
        savename = self.savedir + 'mm_vs_chieff_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.png'
        plt.savefig(savename, dpi=300, bbox_inches='tight', transparent=True)
        logging.info(f"Mismatch vs chi_eff plot saved to {savename}")
        plt.close()

    def plot_mm_vs_spin(self, dfmm):
        """
        Plot the mismatches vs individual spins.
        """
        logging.info("Plotting mismatches vs individual spins.")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for i, (spin_col, ax_row) in enumerate(zip(['spin1z', 'spin2z'], axes)):
            for j, (mm_col, ax) in enumerate(zip(['mismatch_amp', 'mismatch_freq'], ax_row)):
                ax.plot(dfmm[spin_col], dfmm[mm_col], 'o',
                        markersize=3, alpha=0.5, markeredgewidth=0.25, markeredgecolor='black')
                ax.set_xlabel(f'{spin_col}', fontsize=12)
                ax.set_ylabel(f'{mm_col}', fontsize=12)
                ax.set_yscale('log')
                ax.set_title(f'{mm_col} vs {spin_col}', fontsize=12)
        plt.tight_layout()
        savename = self.savedir + 'mm_vs_spin_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.png'
        plt.savefig(savename, dpi=300, bbox_inches='tight', transparent=True)
        logging.info(f"Mismatch vs spin plot saved to {savename}")
        plt.close()

        # Plot mismatch for hplus and hcross vs individual spins
        logging.info("Plotting hplus/hcross mismatches vs individual spins.")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for i, (spin_col, ax_row) in enumerate(zip(['spin1z', 'spin2z'], axes)):
            for j, (mm_col, ax) in enumerate(zip(['mismatch_hplus', 'mismatch_hcross'], ax_row)):
                ax.plot(dfmm[spin_col], dfmm[mm_col], 'o',
                    markersize=3, alpha=0.5, markeredgewidth=0.25, markeredgecolor='black')
                ax.set_xlabel(f'{spin_col}', fontsize=12)
                ax.set_ylabel(f'{mm_col}', fontsize=12)
                ax.set_yscale('log')
                ax.set_title(f'{mm_col} vs {spin_col}', fontsize=12)
        plt.tight_layout()
        savename = self.savedir + 'mm_vs_spin_hphc_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.png'
        plt.savefig(savename, dpi=300, bbox_inches='tight', transparent=True)
        logging.info(f"Mismatch hplus/hcross vs spin plot saved to {savename}")
        plt.close()


    def test_uq(self, batch_size=1, Nruns=1000, plot_hist=True, plotonlyone=False, 
                fontsize=12, labelsize=10):
        """
        Test the uncertainty quantification (UQ) of the model for 1000 random
        sample generation corresponding the same input parameters. The output
        or the error can be visualized as mismatch values for each of these
        generated compared to the actual waveform. Ideally, if our model training
        is perfect, all the mismatches should be the same!

        TODO: Have a dataloader such that it can load specific values from the
        test dataset. Perhaps, need to modify the test dataset classes.

        For batch_size > 1, we will check the mismatch of a random waveform in the batch 
        against the mismatches of all the other waveforms in the batch. This value should 
        be pretty close to zero, although the mismatch values across diff batches is expected
        to differ by a large amount! This was suggested by the reviewer!

        Arguments
        ---------
        batch_size: int, optional
            The batch size to use for loading the test data. Default is 1, since we
            want to test the same input parameters for multiple generations to evaluate UQ.
        Nruns: int, optional
            The number of random generations to perform for the same input parameters. Default is 1000.
        plot_hist: bool, optional
            Whether to plot the histogram of mismatch values for the generated samples. Default is True.
        fontsize: int, optional
            The font size for the plot labels and titles. Default is 12.
        """
        logging.info(f"Testing with model: {self.model_path}")
        # Load the trained model
        preset_array_size = 8190 if args.fcutoff or args.aligned else PRESET_ARRAY_SIZE
        num_classes = 4 if args.aligned else 2
        if self.modeltype=='cae':
            model = CAE(input_shape=(2, preset_array_size), num_classes=num_classes, 
                        key_shape=(2,2)).to(args.device)
        else:
            model = CVAE(input_shape=(2, preset_array_size), num_classes=num_classes, 
                    key_shape=(2,2)).to(args.device)
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        model.to(getattr(torch, self.precision))
        model.to(self.device)
        model.eval()
        logging.info("Model loaded and set to evaluation mode.")

        fig, axes = plt.subplots(1, 2, figsize=(10,5))

        test_loader = self.setdataloader(batch_size=batch_size)
        # `batch_size`=1, so that we can directly send the full batch for test!
        # Check if the specific value exists in labels
        # if the value is not found, then just take the last sample in the batch!
        specific_value = [10, 10, -0.5, 0.5]  # Replace with the desired label value
        if batch_size==1:
            for batch in iter(test_loader):
                x, labels, keys, phases, strains, attr = batch
                if any((labels == torch.tensor(specific_value, device=self.device)).all(dim=1)):
                    idx = (labels == torch.tensor(specific_value, device=self.device)).all(dim=1).nonzero(as_tuple=True)[0].item()
                    x, labels, keys, phases, strains, attr = x[idx], labels[idx], keys[idx], phases[idx], strains[idx], attr
                    print(f"Found specific value {specific_value} in the test set.")
                    break
        else:
            logging.info(f"Batch size is {batch_size}, so taking the first batch for testing UQ. \
                         We will check UQ by comparing mismatch of a random waveform in the batch, against the \
                         mismatches of all the other waveforms in the batch. This value should be pretty close to \
                         zero, although the mismatch values across diff batches is expected to differ by a large amount!")


        mmtot_amp, mmtot_freq = [], []
        mmtot_hplus, mmtot_hcross = [], []
        for i in range(Nruns):

            # If batch_size > 1, choose a batch from the dataloader
            if batch_size > 1:
                try:
                    x, labels, keys, phases, strains, attr = next(iter(test_loader))
                    labels = labels.to(device)
                    # logging.debug(f'Batch {i}: Testing on batch with labels {labels}')
                except StopIteration:
                    logging.warning("Reached end of test dataloader while testing UQ. Restarting dataloader.")
                    test_loader = self.setdataloader(batch_size=batch_size)
                    x, labels, keys, phases, strains, attr = next(iter(test_loader))
                    labels = labels.to(device)
                    # logging.debug(f'Batch {i}: Testing on batch with labels {labels}')

            # Move labels to the appropriate device
            labels = labels.to(device)
            # logging.info(f'Choosing to test sample {labels}')

            with torch.no_grad():
                logging.info(f'Testing run: {i}')
                # Use the label-conditioned encoders and decoder to generate data
                z1_mean, z1_log_var = model.encode_label_for_x(labels)
                z1p_mean, z1p_log_var = model.encode_label_for_key(labels)
                if self.modeltype=='cae':
                    reconst = model.decode(z1_mean, z1p_mean, labels)
                else:
                    z1 = model.reparameterize(z1_mean, z1_log_var)
                    z1p = model.reparameterize(z1p_mean, z1p_log_var)
                    reconst = model.decode(z1, z1p, labels)
                
                if not self.aligned:
                    logging.info('Test for current batch completed. Removing zero padding if any.')
                    x, reconst, phases = removezeros(x, reconst, phases, attr)
                    logging.debug(f'new shapes, Input: {x.shape}, Reconstructed: {reconst.shape}, phases: {phases.shape}')
                
                # NOTE: These returned mismatch arrays constain values for the whole batch!
                mismatch_amp, mismatch_freq, chirpmasses, totalmasses, massratios \
                    = plot_mismatch(x, reconst, labels, keys, savedir=self.savedir, nobatchwiseplot=True)
                logging.debug("Amplitude and Frequency mismatch calculated for current batch.")
                logging.debug("Calculating hplus/hcross mismatch for current batch.")
                mismatch_hplus, mismatch_hcross, chirpmasses, totalmasses, massratios, chieffs, num_saved_overplots \
                    = plot_polarization_mismatch(x, reconst, labels, keys, phases, strains, attr,
                                                savedir=self.savedir, nobatchwiseplot=True,
                                                num_saved_overplots=None)
                
                if batch_size > 1:
                    # Check UQ by comparing mismatch of a random waveform in the batch, against the mismatches of 
                    # all the other waveforms in the batch. We do this by assuming the selected random mismatch to
                    # be the "true" mismatch for that batch, and then calculating the mean of absolute differences of all the 
                    # other mismatches in the batch with this "true" mismatch.
                    # TODO: Check if `flatten` is required here!
                    random_idx = np.random.randint(0, batch_size)
                    random_mismatch_amp = mismatch_amp[random_idx].flatten()[0]
                    random_mismatch_freq = mismatch_freq[random_idx].flatten()[0]
                    random_mismatch_hplus = mismatch_hplus[random_idx].flatten()[0]
                    random_mismatch_hcross = mismatch_hcross[random_idx].flatten()[0]
                    batch_mismatch_amp = mismatch_amp.flatten()
                    batch_mismatch_freq = mismatch_freq.flatten()
                    batch_mismatch_hplus = mismatch_hplus.flatten()
                    batch_mismatch_hcross = mismatch_hcross.flatten()
                    amp_uq = np.mean(np.abs(batch_mismatch_amp - random_mismatch_amp))
                    freq_uq = np.mean(np.abs(batch_mismatch_freq - random_mismatch_freq))
                    hplus_uq = np.mean(np.abs(batch_mismatch_hplus - random_mismatch_hplus))
                    hcross_uq = np.mean(np.abs(batch_mismatch_hcross - random_mismatch_hcross))
                    logging.info(f'Batch {i}: UQ (mean abs diff) for Amplitude Mismatch: {amp_uq:.2e}')
                    logging.info(f'Batch {i}: UQ (mean abs diff) for Frequency Mismatch: {freq_uq:.2e}')
                    logging.info(f'Batch {i}: UQ (mean abs diff) for hplus Mismatch: {hplus_uq:.2e}')
                    logging.info(f'Batch {i}: UQ (mean abs diff) for hcross Mismatch: {hcross_uq:.2e}')
                    # -- append these UQ values to the total mismatch lists, instead of the actual mismatch values, 
                    # -- since we want to evaluate the UQ of the model across different generations for the same input parameters, 
                    # -- rather than the actual mismatch values which are expected to differ across different generations for the 
                    # -- same input parameters due to the stochastic nature of the model! (as suggested by the reviewer)
                    mmtot_amp.append(amp_uq)
                    mmtot_freq.append(freq_uq)
                    mmtot_hplus.append(hplus_uq)
                    mmtot_hcross.append(hcross_uq)
                    if plot_hist:
                        # For histogram, we will plot the final values after all iterations are done!
                        continue
                    else:
                        axes[0].plot(i, amp_uq, 'o', color='blue',
                                markersize=4, markeredgewidth=0.25, markeredgecolor='black')
                        axes[0].plot(i, freq_uq, 'x', color='orange',
                                markersize=4, markeredgewidth=0.25, markeredgecolor='black')
                        axes[1].plot(i, hplus_uq, 'o', color='blue',
                                markersize=4, markeredgewidth=0.25, markeredgecolor='black')
                        # axes[1].plot(i, hcross_uq, 'x', color='orange',
                        #         markersize=4, markeredgewidth=0.25, markeredgecolor='black')
                else:
                    mmtot_amp.append(mismatch_amp.flatten()[0])
                    mmtot_freq.append(mismatch_freq.flatten()[0])
                    mmtot_hplus.append(mismatch_hplus.flatten()[0])
                    mmtot_hcross.append(mismatch_hcross.flatten()[0])
                    axes[0].plot(i, mismatch_amp.flatten(), '.', color='grey',
                            markersize=4, markeredgewidth=0.25, markeredgecolor='black')
                    axes[0].plot(i, mismatch_freq.flatten(), 'x', color='grey',
                            markersize=4, markeredgewidth=0.25, markeredgecolor='black')
                    axes[1].plot(i, mismatch_hplus.flatten(), '.', color='grey',
                            markersize=4, markeredgewidth=0.25, markeredgecolor='black')
                    axes[1].plot(i, mismatch_hcross.flatten(), 'x', color='grey',
                            markersize=4, markeredgewidth=0.25, markeredgecolor='black')

        mu_amp, std_amp = np.mean(mmtot_amp), np.std(mmtot_amp)
        mu_freq, std_freq = np.mean(mmtot_freq), np.std(mmtot_freq)
        mu_hplus, std_hplus = np.mean(mmtot_hplus), np.std(mmtot_hplus)
        mu_hcross, std_hcross = np.mean(mmtot_hcross), np.std(mmtot_hcross)
        print(f'Mean Frequency Mismatch: {mu_freq:.2e} ± {std_freq:.2e}')
        print(f'Mean Amplitude Mismatch: {mu_amp:.2e} ± {std_amp:.2e}')
        print(f'Mean hplus Mismatch: {mu_hplus:.2e} ± {std_hplus:.2e}')
        print(f'Mean hcross Mismatch: {mu_hcross:.2e} ± {std_hcross:.2e}')

        # -- Save mismatch uncertainty values to a dataframe and save as csv file
        dfuq = pd.DataFrame({
            'mmuq_amp': mmtot_amp,
            'mmuq_freq': mmtot_freq,
            'mmuq_hplus': mmtot_hplus,
            'mmuq_hcross': mmtot_hcross,
        })
        savename = f'{self.savedir}/uq-test-' + 'hist-' if plot_hist else ''
        savename += f'mean-abs-diff-{Nruns}-' if batch_size > 1 else ''
        savename += datetime.now().strftime('%Y%m%d_%H%M%S')
        dfuq.to_csv(savename + '.csv', index=False)
        logging.info(f"Mismatch uncertainty values saved to {savename}.csv")

        if batch_size == 1:
            axes[0].legend(['Amplitude', 'Frequency'], loc='upper right')
            label = f'$m_1$={labels[0][0]:.2f}, $m_2$={labels[0][1]:.2f}, $\\chi_1$={labels[0][2]:.2f}, $\\chi_2$={labels[0][3]:.2f}' \
                if self.aligned else f'$m_1$={labels[0][0]:.2f}, $m_2$={labels[0][1]:.2f}'
            axes[0].text(0.05, 0.025, label, transform=axes[0].transAxes, ha='left', fontsize=fontsize)
            axes[1].text(0.05, 0.10, label, transform=axes[1].transAxes, ha='left', fontsize=fontsize)
            axes[0].text(0.05, 0.03,  f'$|\\delta A|$={mu_amp:.2e}' + ', ' +
                        f'$|\\delta f|$={mu_freq:.2e}\n', transform=axes[0].transAxes, ha='left', fontsize=fontsize)
            axes[1].text(0.05, 0.05, '$|\\delta h_{+}|$='+f'{mu_hplus:.2e}' + ', ' +
                        '$|\\delta h_{\\times}|$='+f'{mu_hcross:.2e}', transform=axes[1].transAxes, ha='left', fontsize=fontsize)
            axes[1].legend(['$h_{+}$', '$h_{\\times}$'], loc='upper right')
        elif batch_size > 1 and plot_hist:
            # Plot histogram of mismatch uncertainty values in log scale
            # This is for the final mismatch uncertainty values across all iterations, 
            # which is more meaningful than plotting the histogram for each iteration!
            ampbins = np.logspace(np.log10(dfuq['mmuq_amp'].min())+1e-10, 
                               np.log10(dfuq['mmuq_amp'].max())+1e-10, 50)
            hpbins = np.logspace(np.log10(dfuq['mmuq_hplus'].min())+1e-10,
                                 np.log10(dfuq['mmuq_hplus'].max())+1e-10, 50)
            dfuq.hist('mmuq_amp', bins=ampbins, ax=axes[0], grid=False, edgecolor='black', color='lightblue')
            dfuq.hist('mmuq_hplus', bins=hpbins, ax=axes[1], grid=False, edgecolor='black', color='lightblue')
            # axes[0].text(0.95, 0.85, f'A(t)\nMedian: {np.median(mmtot_amp):.2e}\nMean: {np.mean(mmtot_amp):.2e}', 
            #        transform=axes[0].transAxes, fontsize=fontsize, va='top', ha='right')
            # axes[1].text(0.95, 0.85, '$h_{+}$\n'+f'Median: {np.median(mmtot_hplus):.2e}\nMean: {np.mean(mmtot_hplus):.2e}', 
            #        transform=axes[1].transAxes, fontsize=fontsize, va='top', ha='right')
            # -- overwrite legend with median value information instead!
            axes[1].set_xlim(left=1e-2)
            if not plotonlyone:
                dfuq.hist('mmuq_freq', bins=ampbins, ax=axes[0], grid=False, edgecolor='black', color='salmon', alpha=0.60)
                dfuq.hist('mmuq_hcross', bins=hpbins, ax=axes[1], grid=False, edgecolor='black', color='salmon', alpha=0.60)
                axes[0].legend([f'$A(t)$\nMedian: {np.median(mmtot_amp):.2e}',
                                f'$f(t)$\nMedian: {np.median(mmtot_freq):.2e}'], loc='upper right')
                axes[1].legend([f'$h_+$\nMedian: {np.median(mmtot_hplus):.2e}',
                                '$h_{\\times}$'+f'\nMedian: {np.median(mmtot_hcross):.2e}'], loc='upper right')
            else:
                axes[0].legend([f'$A(t)$\nMedian: {np.median(mmtot_amp):.2e}'], loc='upper left')
                axes[1].legend([f'$h_+$\nMedian: {np.median(mmtot_hplus):.2e}'], loc='upper right')
            for ax in axes:
                ax.set_title(None)
                ax.tick_params(which='both', direction='in', top=True, right=True)
                ax.tick_params(labelsize=labelsize)
                ax.set_xscale('log')
                ax.set_xlabel('Mismatch Uncertainty', fontsize=fontsize)
                ax.set_ylabel('Count', fontsize=fontsize)
                # -- y-axis major ticks should be integer values
                ax.yaxis.set_major_locator(tck.MaxNLocator(integer=True))
        else:
            axes[1].set_ylim(bottom=1e-2)
            axes[0].text(0.05, 0.85,  '$|\\delta A|$='+f'{mu_amp:.2e}'+
                        '\n$|\\delta f|$='+f'{mu_freq:.2e}\n', transform=axes[0].transAxes, ha='left', fontsize=fontsize)
            axes[1].text(0.05, 0.05, '$|\\delta h_{+}|$='+f'{mu_hplus:.2e}', transform=axes[1].transAxes, ha='left', fontsize=fontsize)
            axes[1].legend(['$h_{+}$'], loc='upper right')
            for ax in [axes[0], axes[1]]:
                ax.set_xlabel('Iteration', fontsize=fontsize)
                ax.set_ylabel('Mismatch Uncertainty', fontsize=fontsize)
                ax.set_yscale('log')
                ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
                ax.yaxis.set_minor_locator(tck.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
                ax.tick_params(which='both', direction='in', top=True, right=True)
        plt.tight_layout()
        plt.savefig(savename+'.png', dpi=300, transparent=True)
        plt.savefig(savename+'-white.png', dpi=300)
        plt.show()
        plt.close()
        print("All UQ tests completed.")

    
    def test_uq_iter(self, Nruns=1000, Nwaves=1000, fontsize=12, labelsize=10):
        """
        Test the latent sampling uncertainty of the trained model, by first
        calculating the mean and standard deviation (call mismatch uncertainty) in the mismatch value of a single
        random waveform generated `Nruns` times, and then iterating this process for
        `Nwaves` number of random waveforms. Finally, we will quote the mean and standard
        deviation of the mismatch uncertainty across all these `Nwaves` number of waveforms, 
        which will give us a good idea about the latent sampling uncertainty of the model 
        across different input parameters. This is a more rigorous test of the UQ of the model, 
        compared to the `test_uq` function where we only test the UQ for a single random waveform.
        Lastly, we plot a histogram of the mismatch uncertainty values across all these `Nwaves` number of waveforms,
        which will give us a visual representation of the distribution of the mismatch uncertainty values 
        across different input parameters.

        Since we need to also calculate the polarizations from the reconstructed amplitude and
        frequency series, we need to work with data from the test dataset, from which we would
        select `Nwaves` number of random waveforms, and then for each waveform, we will generate 
        `Nruns` number of reconstructions from the model, and then calculate the mismatch uncertainty 
        for each waveform across these `Nruns` number of reconstructions.

        Arguments
        ---------
        batch_size: int, optional
            The batch size to use for loading the test data. Default is 1, since we want to test the 
            same input parameters for multiple generations to evaluate UQ.
        Nruns: int, optional
            The number of random generations to perform for the same input parameters. Default is 1000.
        Nwaves: int, optional
            The number of random waveforms to test the UQ on. Default is 1000.

        Output
        ------
        A histogram of the mismatch uncertainty values across all the `Nwaves` number of waveforms, 
        and the mean and standard deviation of the mismatch uncertainty values across all these `Nwaves` number
        of waveforms, which will give us a good idea about the latent sampling uncertainty of the model
        across different input parameters.
        """
        logging.info(f"Testing latent sampling uncertainty with model: {self.model_path}")
        logging.info(f"Number of random generations for each waveform (Nruns): {Nruns}")
        logging.info(f"Number of random waveforms to test UQ on (Nwaves): {Nwaves}")

        savename = self.savedir + f'uq-hphc-hist-Nwaves-{Nwaves}-Nruns-{Nruns}'
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        # -- initialize CSV file to data after each Nwave iteration
        csvfname = savename + '-' + now + '.csv'
        with open(csvfname, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['mmuq_amp', 'mmuq_freq', 'mmuq_hplus', 'mmuq_hcross'])

        # Load the trained model
        preset_array_size = 8190 if args.fcutoff or args.aligned else PRESET_ARRAY_SIZE
        num_classes = 4 if args.aligned else 2
        if self.modeltype=='cae':
            model = CAE(input_shape=(2, preset_array_size), num_classes=num_classes, 
                        key_shape=(2,2)).to(args.device)
        else:
            model = CVAE(input_shape=(2, preset_array_size), num_classes=num_classes, 
                    key_shape=(2,2)).to(args.device)
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        model.to(getattr(torch, self.precision))
        model.to(self.device)
        model.eval()
        logging.info("Model loaded and set to evaluation mode.")

        # -- Set up dataloader for test dataset
        test_loader = self.setdataloader(batch_size=1)

        all_mm_means_amp, all_mm_stds_amp, all_mm_means_freq, all_mm_stds_freq = [], [], [], []
        all_mm_means_hp, all_mm_stds_hp, all_mm_means_hc, all_mm_stds_hc = [], [], [], []
        for i in tqdm(range(Nwaves), desc="Nwaves", unit="waveform", nrows=80):

            # Select a random waveform from a random batch from the dataloader
            # NOTE: We only work with batch_size=1 here, since we want to test the same input parameters 
            # for multiple generations to evaluate UQ, and then iterate this process for `Nwaves` number of waveforms.
            try:
                x, labels, keys, phases, strains, attr = next(iter(test_loader))
                labels = labels.to(device)
            except StopIteration:
                logging.warning("Reached end of test dataloader while testing UQ. Restarting dataloader.")
                test_loader = self.setdataloader(batch_size=1)
                x, labels, keys, phases, strains, attr = next(iter(test_loader))
                labels = labels.to(device)

            mm_amp, mm_freq = [], []
            mm_hp, mm_hc = [], []
            for j in tqdm(range(Nruns), desc="Nruns", unit="run", nrows=80):
                with torch.no_grad():
                    z1_mean, z1_log_var = model.encode_label_for_x(labels)
                    z1p_mean, z1p_log_var = model.encode_label_for_key(labels)
                    if self.modeltype=='cae':
                        reconst = model.decode(z1_mean, z1p_mean, labels)
                    else:
                        z1 = model.reparameterize(z1_mean, z1_log_var)
                        z1p = model.reparameterize(z1p_mean, z1p_log_var)
                        reconst = model.decode(z1, z1p, labels)
                    
                    if not self.aligned:
                        logging.info('Test for current batch completed. Removing zero padding if any.')
                        x, reconst, phases = removezeros(x, reconst, phases, attr)
                        logging.debug(f'new shapes, Input: {x.shape}, Reconstructed: {reconst.shape}, phases: {phases.shape}')
                    
                    # NOTE: These returned mismatch arrays constain values for the whole batch!
                    mismatch_amp, mismatch_freq, chirpmasses, totalmasses, massratios \
                        = plot_mismatch(x, reconst, labels, keys, savedir=self.savedir, nobatchwiseplot=True)
                    logging.debug("Amplitude and Frequency mismatch calculated for current batch.")
                    logging.debug("Calculating hplus/hcross mismatch for current batch.")
                    mismatch_hplus, mismatch_hcross, chirpmasses, totalmasses, massratios, chieffs, num_saved_overplots \
                        = plot_polarization_mismatch(x, reconst, labels, keys, phases, strains, attr,
                                                    savedir=self.savedir, nobatchwiseplot=True,
                                                    num_saved_overplots=None)
                    mm_amp.append(mismatch_amp.flatten()[0])
                    mm_freq.append(mismatch_freq.flatten()[0])
                    mm_hp.append(mismatch_hplus.flatten()[0])
                    mm_hc.append(mismatch_hcross.flatten()[0])
            # -- Calculate mean and std of mismatch values across `Nruns` number of generations 
            # for the current waveform, and append to the total lists
            all_mm_means_amp.append(np.mean(mm_amp))
            all_mm_stds_amp.append(np.std(mm_amp))
            all_mm_means_freq.append(np.mean(mm_freq))
            all_mm_stds_freq.append(np.std(mm_freq))
            all_mm_means_hp.append(np.mean(mm_hp))
            all_mm_stds_hp.append(np.std(mm_hp))
            all_mm_means_hc.append(np.mean(mm_hc))
            all_mm_stds_hc.append(np.std(mm_hc))
            with open(csvfname, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([np.std(mm_amp), np.std(mm_freq), np.std(mm_hp), np.std(mm_hc)])
            logging.info(f'Completed UQ for waveform {i+1}/{Nwaves}. Mean and Std of hplus Mismatch: {np.mean(mm_hp):.2e} ± {np.std(mm_hp):.2e}, Mean and Std of hcross Mismatch: {np.mean(mm_hc):.2e} ± {np.std(mm_hc):.2e}')
                         
        # -- Plot histogram of the standard deviation of the mismatch values across all 
        # `Nwaves` number of waveforms, which we call the mismatch uncertainty, to evaluate the 
        # latent sampling uncertainty of the model across different input parameters.
        fig, axes = plt.subplots(1, 2, figsize=(10,5))
        dfuq = pd.DataFrame({
            'mmuq_amp': all_mm_stds_amp,
            'mmuq_freq': all_mm_stds_freq,
            'mmuq_hplus': all_mm_stds_hp,
            'mmuq_hcross': all_mm_stds_hc,
        })
        hpbins = np.logspace(np.log10(dfuq['mmuq_hplus'].min())+1e-10,
                             np.log10(dfuq['mmuq_hplus'].max())+1e-10, 50)
        dfuq.hist('mmuq_hplus', bins=hpbins, ax=axes[0], grid=False, edgecolor='black', color='lightblue')
        dfuq.hist('mmuq_hcross', bins=hpbins, ax=axes[1], grid=False, edgecolor='black', color='lightblue')
        types = ['mmuq_hplus', 'mmuq_hcross']
        titles = [ '$\\mathbf{h_{+}}$', '$\\mathbf{h_{\\times}}$']
        for i in range(len(types)):
            ax = axes[i]
            ax.set_xlim(1e-5, 1e-1)
            xloc, yloc, ha = 0.05, 0.95, 'left'
            ax.set_xscale('log')
            ax.tick_params(which="both", direction='in', top=True, right=True)
            ax.tick_params(labelsize=labelsize)
            ax.set_xlabel('Mismatch Uncertainty', fontsize=fontsize)
            ax.set_ylabel('Count', fontsize=fontsize)
            ax.text(xloc, yloc, titles[i], fontweight='bold',
                    transform=ax.transAxes, fontsize=labelsize, va='top', ha=ha)
            ax.text(xloc, yloc - 0.1, f'Mode: {dfuq[types[i]].mode()[0]:.2e}\nMean: {dfuq[types[i]].mean():.2e}\nMedian: {dfuq[types[i]].median():.2e}',
                    transform=ax.transAxes, fontsize=labelsize, va='top', ha=ha)
            ax.set_title(None)
            ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
        plt.tight_layout()
        plt.savefig(savename+'-'+now+'.png', dpi=300, bbox_inches='tight', transparent=True)
        plt.savefig(savename+'-white'+'-'+now+'.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        logging.info("All UQ iterations completed. Plotting histogram of mismatch uncertainty values across all waveforms.")


    def test_timecomplexity(self, num=int(1e6), num_start=int(1e4)):
        """
        Test the time complexity of the model for generating a 1-10e4 ish number of samples.
        This is useful for understanding the efficiency of the model in real-time
        applications.

        TODO,BUG: We should include the time it takes to convert the generated amplitude and frequency series to hplus and hcross polarizations, since this is a necessary step for any real-time application of the model, and it can be a bottleneck in the generation process. We can use the `plot_polarization_mismatch` function to calculate the polarizations from the generated amplitude and frequency series, and then include this time in our time complexity calculation.
        """
        # Nruns = [1, 10, 50, 100, 500, 1e3, 5e3, 1e4]
        # Nruns = np.logspace(0, 5, num=num, dtype=int)
        # Nruns = [int(n) for n in [1, 10, 50, 100, 500, 1e3, 5e3, 1e4, 5e4]]
        Nruns = np.arange(num_start,num+1)
        logging.info(f"Testing with model: {self.model_path}")
        # Load the trained model
        preset_array_size = 8190 if args.fcutoff or args.aligned else PRESET_ARRAY_SIZE
        num_classes = 4 if args.aligned else 2
        if self.modeltype=='cae':
            model = CAE(input_shape=(2, preset_array_size), num_classes=num_classes, 
                        key_shape=(2,2)).to(self.device)
        else:
            model = CVAE(input_shape=(2, preset_array_size), num_classes=num_classes, 
                        key_shape=(2,2)).to(self.device)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        model.to(getattr(torch, self.precision))
        model.eval()
        logging.info("Model loaded and set to evaluation mode.")

        # -- Open CSV file to save results on the go
        csv_fname = self.savedir + 'timecomplexity_results-' + str(self.device) + '-' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv'
        csvfile = open(csv_fname, mode='w', newline='')
        csvfile.write('num_samples,time_seconds\n')  # Write header row
        logging.info(f"CSV file opened for writing time complexity results: {csv_fname}")

        # -- create some dummy warm-up runs!
        # NOTE: To remove the cold-start time from the calculation,
        # which happens on the first time the model is called. CUDA
        # will initialize a location on the GPU to store the model,
        # and setup the necessary instruction sets during the first time.
        # After the warm-up, the same GPU location is used, so the initial
        # warm-up time is not required. Plus, till all 100% of the GPU
        # memory is being used, the time taken for N waveforms to generate
        # is mostly the same. Changes occur after GPU memory is filled-up!
        for _ in range(10):
            labels = torch.tensor([[10, 10, -0.5, 0.5],[10, 10, -0.5, 0.5],[10, 10, -0.5, 0.5]], 
                                  dtype=getattr(torch, self.precision)).to(self.device)
            with torch.no_grad():
                z1_mean, z1_log_var = model.encode_label_for_x(labels)
                z1p_mean, z1p_log_var = model.encode_label_for_key(labels)
                if self.modeltype=='cae':
                    _ = model.decode(z1_mean, z1p_mean, labels)
                else:
                    z1 = model.reparameterize(z1_mean, z1_log_var)
                    z1p = model.reparameterize(z1p_mean, z1p_log_var)
                    _ = model.decode(z1, z1p, labels)
        logging.info("Completed warm-up runs to mitigate cold-start time.")


        times = []
        for Nr in Nruns:
            # Generate random labels within the training range
            m1 = np.random.uniform(5, 75, Nr)
            m2 = np.random.uniform(5, 75, Nr)
            if self.aligned:
                spin1z = np.random.uniform(-0.9, 0.9, Nr)
                spin2z = np.random.uniform(-0.9, 0.9, Nr)
                labels = np.vstack((m1, m2, spin1z, spin2z)).T
            else:
                labels = np.vstack((m1, m2)).T
            labels = torch.tensor(labels, dtype=getattr(torch, self.precision)).to(self.device)
            logging.info(f'Choosing to test sample size {labels.shape}')

            # Measure time taken by model to generate samples
            if self.device == 'cuda':
                torch.cuda.synchronize() # Wait for warm-up to finish
            start_time = time.time()
            with torch.no_grad():
                # Use the label-conditioned encoders and decoder to generate data
                z1_mean, z1_log_var = model.encode_label_for_x(labels)
                z1p_mean, z1p_log_var = model.encode_label_for_key(labels)
                if self.modeltype=='cae':
                    reconst = model.decode(z1_mean, z1p_mean, labels)
                else:
                    z1 = model.reparameterize(z1_mean, z1_log_var)
                    z1p = model.reparameterize(z1p_mean, z1p_log_var)
                    reconst = model.decode(z1, z1p, labels)
            if self.device == 'cuda':
                torch.cuda.synchronize() # Wait for warm-up to finish
            end_time = time.time()
            elapsed_time = end_time - start_time
            times.append(elapsed_time)
            logging.info(f'Time taken to generate {Nr} samples: {elapsed_time:.4f} seconds')
            # -- Write the result to CSV file each time, so that if the process is interrupted, 
            # we still have the results up to that point.
            csvfile.write(f'{Nr},{elapsed_time}\n')

            # -- Flush out memory storage after each run, to avoid pileup of memory and consequent slowdown 
            # in time taken for generation of samples in later runs.
            # Otherwise, around 2x10^4 waveforms, the time taken for generation approachs the vertical asymptote, 
            # which is not expected for a well-behaved model! This is likely due to the GPU memory getting filled 
            # up and causing slowdown in generation of samples.
            torch.cuda.empty_cache()
            logging.debug("End of run. Emptied CUDA cache to prevent memory pileup.")

        # Close the CSV file after writing all results
        csvfile.close()
        logging.info(f"Time complexity results saved to CSV file: {csvfile.name}")

        # # Save the time complexity results to a CSV file
        # df_time = pd.DataFrame({'num_samples': Nruns, 'time_seconds': times})
        # df_time.to_csv(self.savedir + 'timecomplexity_results-' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv', index=False)

        # Plot the time taken v/s number of samples plots
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(Nruns, times, 'o', color='grey', markersize=6,
                markeredgewidth=0.25, markeredgecolor='black')
        ax.set_xlabel('Number of Samples', fontsize=12)
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.xaxis.set_minor_locator(tck.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
        ax.yaxis.set_minor_locator(tck.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.text(0.05, 0.95, f'N={len(Nruns)}', transform=ax.transAxes, fontsize=10, verticalalignment='top')
        plt.tight_layout()
        figname = f'{self.savedir}/timecomplexity-test-' + datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(figname+'.png', dpi=300, transparent=True)
        plt.savefig(figname+'-white.png', dpi=300)
        plt.close()
        logging.info("Time complexity test completed and plot saved.")


    def test_timecomplexity_compare(self, iters=100):
        """
        Test the time complexity of the model for generating a 1-10e4 ish number of samples.
        This is useful for understanding the efficiency of the model in real-time
        applications.
        """
        Nruns = np.arange(1,iters+1)
        np.random.shuffle(Nruns)
        logging.info(f"Using {Nruns=}")

        # Load the trained model
        logging.info(f"Testing with model: {self.model_path}")
        preset_array_size = 8190 if args.fcutoff or self.aligned else PRESET_ARRAY_SIZE
        num_classes = 4 if args.aligned else 2
        if self.modeltype=='cae':
            model = CAE(input_shape=(2, preset_array_size), num_classes=num_classes, 
                        key_shape=(2,2)).to(args.device)
        else:
            model = CVAE(input_shape=(2, preset_array_size), num_classes=num_classes, 
                    key_shape=(2,2)).to(args.device)
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        model.to(torch.float64)
        model.to(device)
        model.eval()
        logging.info("Model loaded and set to evaluation mode.")

        modeltimes, basetimes = [], []
        massratios, chieffs = [], []
        romtimes, opttimes = [], []
        for Nr in tqdm(Nruns, ncols=100):
            # Generate random labels within the training range
            m1 = np.random.uniform(5, 75, Nr)
            q = np.random.uniform(1, 10, Nr)
            m2 = m1 / q
            if self.aligned:
                spin1z = np.random.uniform(-0.9, 0.9, Nr)
                spin2z = np.random.uniform(-0.9, 0.9, Nr)
                labels = np.vstack((m1, m2, spin1z, spin2z)).T
            else:
                labels = np.vstack((m1, m2)).T
            labels = torch.tensor(labels, dtype=torch.float64).to(device)
            logging.info(f'Choosing to test sample size {labels.shape}')

            massratios = q
            chieffs = (m1 * spin1z + m2 * spin2z) / (m1 + m2) if self.aligned else np.zeros(Nr)
            
            # Measure time taken by model to generate samples
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
            modeltimes.append(elapsed_time)
            logging.info(f'Time taken to generate {Nr} samples: {elapsed_time:.4f} seconds')

            # Measure base time taken to generate random samples
            waveform_kwargs = {}
            start_time = time.time()
            for i in range(Nr):
                waveform_kwargs['mass1'] = m1[i]
                waveform_kwargs['mass2'] = m2[i]
                if self.aligned:
                    waveform_kwargs['spin1z'] = spin1z[i]
                    waveform_kwargs['spin2z'] = spin2z[i]
                waveform_kwargs.update({
                    'approximant': self.approximant,
                    'delta_t': DELTA_T,
                    'f_lower': 20.0,  # fix this at 20 Hz
                })
                hp, hc = pycbc.waveform.get_td_waveform(**waveform_kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            basetimes.append(elapsed_time)
            logging.info(f'Base time taken to generate {Nr} samples: {elapsed_time:.4f} seconds')

            # Compare ROM approximant genration times
            waveform_kwargs['approximant'] = 'SEOBNRv4_ROM' # or 'SEOBNRv4ROM'
            start_time = time.time()
            for i in range(Nr):
                waveform_kwargs['mass1'] = m1[i]
                waveform_kwargs['mass2'] = m2[i]
                if self.aligned:
                    waveform_kwargs['spin1z'] = spin1z[i]
                    waveform_kwargs['spin2z'] = spin2z[i]
                waveform_kwargs.update({
                    'approximant': self.approximant,
                    'delta_t': DELTA_T,
                    'f_lower': 20.0,  # fix this at 20 Hz
                })
                print(waveform_kwargs)
                hp, hc = pycbc.waveform.get_td_waveform(**waveform_kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            romtimes.append(elapsed_time)
            logging.info(f'ROM time taken to generate {Nr} samples: {elapsed_time:.4f} seconds')

            # Compare SEOBNRv4opt approximant genration times
            waveform_kwargs['approximant'] = 'SEOBNRv4_opt'  # or 'SEOBNRv4Opt'
            start_time = time.time()
            for i in range(Nr):
                waveform_kwargs['mass1'] = m1[i]
                waveform_kwargs['mass2'] = m2[i]
                if self.aligned:
                    waveform_kwargs['spin1z'] = spin1z[i]
                    waveform_kwargs['spin2z'] = spin2z[i]
                waveform_kwargs.update({
                    'approximant': self.approximant,
                    'delta_t': DELTA_T,
                    'f_lower': 20.0,  # fix this at 20 Hz
                })
                print(waveform_kwargs)
                hp, hc = pycbc.waveform.get_td_waveform(**waveform_kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            opttimes.append(elapsed_time)
            logging.info(f'Optimized base time taken to generate {Nr} samples: {elapsed_time:.4f} seconds')


        # Save data to csv file
        logging.info(f"Nruns shape: {np.shape(Nruns)}, modeltimes shape: {np.shape(modeltimes)}, basetimes shape: {np.shape(basetimes)}, romtimes shape: {np.shape(romtimes)}, massratios shape: {np.shape(massratios)}, chieffs shape: {np.shape(chieffs)}")
        df_time = pd.DataFrame({
            'Nruns': Nruns,
            'model_time': modeltimes,
            'base_time': basetimes,
            'rom_time': romtimes,
            'opt_time': opttimes,
            # 'mass_ratio': massratios, # TODO
            # 'chi_eff': chieffs
        })
        csvname = self.savedir + 'timecomplexity_compare_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv'
        df_time.to_csv(csvname, index=False)
        logging.info(f"Time complexity comparison data saved to {csvname}")

        # Plot time taken comparison between model, base, and ROM
        logging.info("Plotting time complexity comparison between model, base, and ROM.")
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(Nruns, basetimes, '.', color='black', markersize=10,
                markeredgewidth=0.5, markeredgecolor='black')
        ax.plot(Nruns, modeltimes, '*', color='blue', markersize=6,
                markeredgewidth=0.5, markeredgecolor='black')
        ax.plot(Nruns, romtimes, '^', color='red', markersize=6,
                markeredgewidth=0.5, markeredgecolor='black')
        ax.plot(Nruns, opttimes, 'v', color='green', markersize=6,
                markeredgewidth=0.5, markeredgecolor='black')
        ax.legend([self.approximant+'-base', self.approximant+'-ml', 'ROM', 'opt'], loc='upper left')
        ax.set_xlabel('Number of Waveforms Generated', fontsize=12)
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.xaxis.set_minor_locator(tck.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
        ax.yaxis.set_minor_locator(tck.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
        ax.tick_params(which='both', direction='in', top=True, right=True)
        plt.tight_layout()
        figname = f'{self.savedir}/timecomplexity-compare-' + f'{self.device}-' + datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(figname+'.png', dpi=300, transparent=True)
        plt.savefig(figname+'-white.png', dpi=300)
        plt.close()
        logging.info("Time complexity comparison plot saved.")

        # # TODO: This should be done for each `Nrun` or for one of them!
        # # Plot bar plot for avg time taken in each q / chi bins
        # q_bins = np.linspace(np.min(massratios), np.max(massratios), 11)
        # chi_bins = np.linspace(np.min(chieffs), np.max(chieffs), 7)

        # # Digitize massratios and chieffs into bins
        # q_idx = np.digitize(massratios, q_bins) - 1
        # chi_idx = np.digitize(chieffs, chi_bins) - 1

        # # Compute average times per bin
        # avg_modeltimes_q = [np.mean([modeltimes[i] for i in range(len(q_idx)) if q_idx[i] == b])
        #             for b in range(len(q_bins)-1)]
        # avg_basetimes_q = [np.mean([basetimes[i] for i in range(len(q_idx)) if q_idx[i] == b])
        #            for b in range(len(q_bins)-1)]
        # avg_modeltimes_chi = [np.mean([modeltimes[i] for i in range(len(chi_idx)) if chi_idx[i] == b])
        #               for b in range(len(chi_bins)-1)]
        # avg_basetimes_chi = [np.mean([basetimes[i] for i in range(len(chi_idx)) if chi_idx[i] == b])
        #              for b in range(len(chi_bins)-1)]

        # # Plot bar plots for q bins
        # fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        # width = 0.35
        # bin_centers_q = 0.5 * (q_bins[:-1] + q_bins[1:])
        # ax.bar(bin_centers_q - width/2, avg_modeltimes_q, width, label='ML model')
        # ax.bar(bin_centers_q + width/2, avg_basetimes_q, width, label='Base')
        # ax.set_xlabel('Mass Ratio (q)', fontsize=12)
        # ax.set_ylabel('Avg Time (seconds)', fontsize=12)
        # ax.set_yscale('log')
        # ax.legend()
        # plt.tight_layout()
        # plt.savefig(f'{self.savedir}/avg_time_qbins_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', dpi=300)
        # plt.close()

        # # Plot bar plots for chi_eff bins
        # fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        # bin_centers_chi = 0.5 * (chi_bins[:-1] + chi_bins[1:])
        # ax.bar(bin_centers_chi - width/2, avg_modeltimes_chi, width, label='ML model')
        # ax.bar(bin_centers_chi + width/2, avg_basetimes_chi, width, label='Base')
        # ax.set_xlabel('$\\chi_{eff}$', fontsize=12)
        # ax.set_ylabel('Avg Time (seconds)', fontsize=12)
        # ax.set_yscale('log')
        # ax.legend()
        # plt.tight_layout()
        # plt.savefig(f'{self.savedir}/avg_time_chibins_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', dpi=300)
        # plt.close()

        # # Plot time taken for different q / chi-eff bins
        # for times, label in zip([modeltimes, basetimes], 
        #                          ['ML', 'Base']):
        #     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        #     axes[0].plot(massratios, times, 'o', label=label, color='grey', 
        #                markersize=6, markeredgewidth=0.25, markeredgecolor='black')
        #     axes[1].plot(chieffs, times, 'o', label=label, color='grey', 
        #                markersize=6, markeredgewidth=0.25, markeredgecolor='black')
        #     axes[1].set_xlabel('$\\chi_{eff}$', fontsize=12)
        #     axes[0].set_xlabel('Mass Ratio (q)', fontsize=12)
        #     for ax in axes:
        #         ax.tick_params(which='both', direction='in', top=True, right=True)
        #         ax.set_ylabel('Time (seconds)', fontsize=12)
        #         ax.set_yscale('log')
        #         ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
        #         ax.yaxis.set_minor_locator(tck.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
        #         ax.tick_params(which='both', direction='in', top=True, right=True)
        #     plt.tight_layout()
        #     figname = f'{self.savedir}/timecomplexity-distro-' + label
        #     figname += '-' + datetime.now().strftime('%Y%m%d_%H%M%S')
        #     plt.savefig(figname+'.png', dpi=300, transparent=True)
        #     plt.savefig(figname+'-white.png', dpi=300)
        #     plt.close()

    def plot_time_complexity_compare(self, fname=None):
        """
        Plot the time complexity comparison from a saved csv file.
        """
        if fname is None:
            raise NotImplementedError("Please provide the filename of the saved csv file for plotting.")
        logging.info(f"Plotting time complexity comparison from file: {fname}")
        df = pd.read_csv(self.savedir+fname+'.csv')
        # logging.info(df.describe().to_string())
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(df['Nruns'], df['base_time'], '.', color='grey', markersize=10,
                markeredgewidth=0.5, markeredgecolor='black')
        ax.plot(df['Nruns'], df['model_time'], '*', color='grey', markersize=6,
                markeredgewidth=0.5, markeredgecolor='black')
        ax.legend([self.approximant+'-base', self.approximant+'-ml'], loc='upper left')
        ax.set_xlabel('Number of Waveforms Generated', fontsize=12)
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True)
        ax.xaxis.set_minor_locator(tck.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
        ax.yaxis.set_minor_locator(tck.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
        ax.tick_params(which='both', direction='in', top=True, right=True)
        plt.tight_layout()
        figname = fname.replace('.csv', '.png')
        plt.savefig(self.savedir+figname, dpi=300, transparent=True)
        plt.close()
        logging.info("Time complexity comparison plot saved.")


    def generate(self, num_samples=1, labels=None, nomismatch=False):
        """
        Generate new waveform samples using the trained CVAE model.
        If labels are provided, generate samples conditioned on those labels.
        Otherwise, generate samples by sampling from the latent space.

        Parameters:
        -----------
        num_samples : int
            Number of samples to generate if labels are not provided.
        labels : torch.Tensor, optional
            Labels to condition the generation on. Shape should be (num_samples, num_classes).
        nomm : bool
            If True, skip mismatch calculation and plotting.
            
        Returns:
        --------
        generated_samples : torch.Tensor
            Generated waveform samples. Shape is (num_samples, 2, PRESET_ARRAY_SIZE).
        """
        logging.info(f"Generating new samples with model: {self.model_path}")
        # Load the trained model
        preset_array_size = 8190 if args.fcutoff else PRESET_ARRAY_SIZE
        num_classes = 4 if args.aligned else 2
        if self.modeltype=='cae':
            model = CAE(input_shape=(2, preset_array_size), num_classes=num_classes, 
                        key_shape=(2,2))
        else:
            model = CVAE(input_shape=(2, preset_array_size), num_classes=num_classes, 
                    key_shape=(2,2))
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        model.to(torch.float64)
        model.to(device)
        model.eval()
        logging.info("Model loaded and set to evaluation mode.")

        if labels is not None:
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.float64)

            labels = labels.to(device)
            num_samples = labels.shape[0]
            logging.info(f'Generating {num_samples} samples conditioned on provided labels.')
            with torch.no_grad():
                z1_mean, z1_log_var = model.encode_label_for_x(labels)
                z1p_mean, z1p_log_var = model.encode_label_for_key(labels)
                if self.modeltype=='cae':
                    generated = model.decode(z1_mean, z1p_mean, labels)
                else:
                    z1 = model.reparameterize(z1_mean, z1_log_var)
                    z1p = model.reparameterize(z1p_mean, z1p_log_var)
                    generated = model.decode(z1, z1p, labels)
        else:
            logging.info(f'Generating {num_samples} samples by sampling from latent space.')
            with torch.no_grad():
                z1 = torch.randn(num_samples, model.latent_dim_x).to(device)
                z1p = torch.randn(num_samples, model.latent_dim_key).to(device)
                if self.aligned:
                    random_labels = torch.tensor(np.random.uniform(
                        [5, 5, -0.9, -0.9], [75, 75, 0.9, 0.9], size=(num_samples, 4)),
                        dtype=torch.float64).to(device)
                else:
                    random_labels = torch.tensor(np.random.uniform(
                        [5, 5], [75, 75], size=(num_samples, 2)),
                        dtype=torch.float64).to(device)
                generated = model.decode(z1, z1p, random_labels)
        
        if nomismatch:
            return generated

        if not self.aligned:
            raise NotImplementedError("Mismatch calculation for non-aligned spins is not implemented in generate().")
        
        # If nomm is False, calculate and plot mismatches
        logging.info("Calculating mismatches for generated samples.")
        original = np.zeros((num_samples, 2, preset_array_size))
        keys = np.zeros((num_samples, 2, 2))
        phases = np.zeros((num_samples, preset_array_size))
        waves = SEOBNRv4(preset_array_size=preset_array_size+1)
        for i, label in enumerate(labels):
            m1, m2, s1, s2 = label
            hp, hc, amp, phase, freq, _ = waves.get_aligned_vals(m1, m2, s1, s2)
            # If amp is longer than preset_array_size, truncate the first value(s)
            if amp.shape[0] > preset_array_size:
                amp = amp[-preset_array_size:]
                phase = phase[-preset_array_size:]
            original[i, 0, :] = amp
            original[i, 1, :] = freq
            phases[i, :] = phase
            # Store normalization keys
            amp_mean, amp_std = np.mean(amp), np.std(amp)
            freq_mean, freq_std = np.mean(freq), np.std(freq)
            keys[i, 0, :] = [amp_mean, amp_std]
            keys[i, 1, :] = [freq_mean, freq_std]

        # mismatch_amp, mismatch_freq, chirpmasses, totalmasses, massratios \
        #     = plot_mismatch(original, generated, labels, keys,
        #                     savedir=self.savedir, nobatchwiseplot=False)
        logging.info("Amplitude and Frequency mismatch calculated for generated samples.")
        logging.info("Calculating hplus/hcross mismatch for generated samples.")
        mismatch_hplus, mismatch_hcross, chirpmasses, totalmasses, massratios, chieffs, num_saved_overplots \
            = plot_polarization_mismatch(original, generated, labels, keys, phases,
                                        savedir=self.savedir, nobatchwiseplot=False,
                                        num_saved_overplots=0, generating=True)
        return generated

    def test_mismatch_compare(self, test_ml_model=False, get_new_orig_wave=False):
        """
        Test the mismatch of ML-generated, ROM and optimized SEOBNRv4 waveforms 
        against the base SEOBNRv4 waveforms for the same Test dataset.
        This is useful for understanding the accuracy of the ML model in comparison
        to the base, ROM and optimized models across a range of parameters.

        The waveform keyargs are read from the HDF file containing the test dataset, 
        and the same keyargs are used to generate the base, ROM, optimized, and ML 
        waveforms for comparison. These saved attributes include the masses, spins, 
        DELTA_T, f_lower, and approximant used for generating the waveforms in the test dataset.
        """
        if test_ml_model:
            raise NotImplementedError("Mismatch comparison for ML-generated waveforms is not implemented yet.")

        # -- load data from testdataset with batchsize=1, so that we can generate waveforms
        # for each waveform and compare mismatches.
        test_loader = self.setdataloader(batch_size=1)

        # -- open file to save mismatch comparison results
        csv_fname = self.savedir + 'mismatch_comparison_results-' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv'
        csvfile = open(csv_fname, mode='w', newline='')
        csvfile.write('m1,m2,s1,s2,delta_t,f_lower,mm_rom_hp,mm_rom_hc,mm_opt_hp,mm_opt_hc\n')  # Write header row
        logging.info(f"CSV file opened for writing mismatch comparison results: {csv_fname}")

        for (x, labels, keys, phases, strains, attr) in tqdm(iter(test_loader)):
            m1, m2, s1, s2 = labels[0].cpu().numpy()
            delta_t = attr['delta_t'][0]
            f_lower = attr['f_lower'][0]
            logging.info(f"Testing mismatch comparison for waveform with parameters: \
                         m1={m1}, m2={m2}, s1={s1}, s2={s2}, delta_t={delta_t}, f_lower={f_lower}")

            # -- set waveform generation parameters
            wfkwargs = {
                'mass1': m1,
                'mass2': m2,
                'spin1z': s1,
                'spin2z': s2,
                'delta_t': delta_t,
                'f_lower': f_lower,
                'approximant': self.approximant
            }

            # -- get original base waveforms
            hp_hdf = strains[0][0].detach().cpu().numpy()
            hc_hdf = strains[0][1].detach().cpu().numpy()
            logging.info(f'length of hdf hp and hc: {len(hp_hdf)}, {len(hc_hdf)}')
            if get_new_orig_wave:
                hp_orig, hc_orig = pycbc.waveform.get_td_waveform(**wfkwargs)
                hp_orig = hp_orig.trim_zeros()
                hc_orig = hc_orig.trim_zeros()
                logging.info(f'length of original hp and hc after trimming zeros: {len(hp_orig)}, {len(hc_orig)}')
                if len(hp_orig)>PRESET_ARRAY_SIZE:
                    diff = len(hp_orig) - PRESET_ARRAY_SIZE
                    hp_orig = hp_orig[diff:]
                    hc_orig = hc_orig[diff:]
                    logging.info(f'Trimmed hdf hp and hc to preset array size: {len(hp_hdf)}, {len(hc_hdf)}')
                if not args.nosave:
                    plot_hphc_overplot(hp_orig, hc_orig, hp_hdf, hc_hdf, savename='../results/overplot-hdf', label=labels[0], 
                                       transparent=False)
            else:
                hp_orig = hp_hdf
                hc_orig = hc_hdf  # -- plot waveforms to see how they look like
            
            # -- get SEOBNRv4_ROM waveforms (this is frequency-domain)
            wfkwargs['approximant'] = 'SEOBNRv4_ROM'
            hp_rom, hc_rom = pycbc.waveform.get_td_waveform(**wfkwargs)
            logging.info(f'{type(hp_rom)=}, {type(hc_rom)=}, {hp_rom.shape=}, {hc_rom.shape=}')
            hp_rom = hp_rom.trim_zeros()
            hc_rom = hc_rom.trim_zeros()
            hp_rom = np.asarray(hp_rom, dtype=np.float64)
            hc_rom = np.asarray(hc_rom, dtype=np.float64)
            logging.info(f"Original waveform length: {len(hp_orig)}, ROM waveform length: {len(hp_rom)}")
            if len(hp_orig)<len(hp_rom):
                # Align merger points and extract matching segment
                merger_idx_orig = np.argmax(np.abs(hp_orig))
                merger_idx_rom = np.argmax(np.abs(hp_rom))
                
                # Calculate left and right portions relative to merger in original
                left_len = merger_idx_orig
                right_len = len(hp_orig) - merger_idx_orig - 1
                
                # Extract ROM segment centered at its merger with same left/right lengths
                rom_start = max(0, merger_idx_rom - left_len)
                rom_end = min(len(hp_rom), merger_idx_rom + right_len + 1)
                
                # Adjust if we hit boundaries
                if rom_start == 0:
                    rom_end = min(len(hp_rom), left_len + right_len + 1)
                elif rom_end == len(hp_rom):
                    rom_start = max(0, len(hp_rom) - left_len - right_len - 1)
                
                hp_rom = hp_rom[rom_start:rom_end]
                hc_rom = hc_rom[rom_start:rom_end]
                logging.info(f"Aligned ROM waveform at merger. Original length: {len(hp_orig)}, ROM segment length: {len(hp_rom)}")
            assert len(hp_orig) == len(hp_rom), "After alignment, original and ROM waveforms should have the same length."
            assert len(hc_orig) == len(hc_rom), "After alignment, original and ROM waveforms should have the same length."

            # -- plot waveforms to see how they look like
            if not args.nosave:
                plot_hphc_overplot(hp_orig, hc_orig, hp_rom, hc_rom, savename='../results/overplot-rom', label=labels[0], 
                                   transparent=False)
               
            # -- calculate mismatches
            mm_rom_hp = calc_polarization_mismatch(hp_orig, hp_rom, delta_t=delta_t, f_lower=f_lower)
            mm_rom_hc = calc_polarization_mismatch(hc_orig, hc_rom, delta_t=delta_t, f_lower=f_lower)

            # -- get SEOBNRv4_opt waveforms (this is time-domain)
            wfkwargs['approximant'] = 'SEOBNRv4_opt'
            hp_opt, hc_opt = pycbc.waveform.get_td_waveform(**wfkwargs)
            hp_opt = hp_opt.trim_zeros()
            hc_opt = hc_opt.trim_zeros()
            hp_opt = np.asarray(hp_opt, dtype=np.float64)
            hc_opt = np.asarray(hc_opt, dtype=np.float64)
            logging.info(f"Original waveform length: {len(hp_orig)}, Optimized waveform length: {len(hp_opt)}")
            if len(hp_orig)<len(hp_opt):
                hp_opt = hp_opt[-len(hp_orig):]
                hc_opt = hc_opt[-len(hc_orig):]
                logging.info(f"Trimmed Optimized waveform to match original length: {len(hp_opt)}")
            if len(hp_orig)>len(hp_opt):
                hp_orig = hp_orig[-len(hp_opt):]
                hc_orig = hc_orig[-len(hc_opt):]
                logging.info(f"Trimmed original waveform to match Optimized length: {len(hp_orig)}")
            assert len(hp_orig) == len(hp_opt), "After trimming, original and Optimized waveforms should have the same length."
            assert len(hc_orig) == len(hc_opt), "After trimming, original and Optimized waveforms should have the same length."

            # -- plot waveforms to see how they look like
            if not args.nosave:
                plot_hphc_overplot(hp_orig, hc_orig, hp_opt, hc_opt, savename='../results/overplot-opt', label=labels[0], transparent=False)

            # -- calculate mismatches
            mm_opt_hp = calc_polarization_mismatch(hp_orig, hp_opt, delta_t=delta_t, f_lower=f_lower)
            mm_opt_hc = calc_polarization_mismatch(hc_orig, hc_opt, delta_t=delta_t, f_lower=f_lower)
            
            logging.info(f"Calculated mismatches: \
                         ROM hp mismatch={mm_rom_hp:.4e}, ROM hc mismatch={mm_rom_hc:.4e}, \
                         Optimized hp mismatch={mm_opt_hp:.4e}, Optimized hc mismatch={mm_opt_hc:.4e}")

            # -- write results to csv file
            csvfile.write(f'{m1},{m2},{s1},{s2},{delta_t},{f_lower},{mm_rom_hp},{mm_rom_hc},{mm_opt_hp},{mm_opt_hc}\n')
            logging.info("Mismatch comparison results written to CSV file.")


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
        label = labels[i].reshape([labels.shape[1]])
        logging.debug(f"Type of reshaped label: {type(label)}, Shape: {label.shape}")
        
        # NOTE: The overplot waveforms are not normalized!
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
        
        axes[0].plot(np.arange(len(orig_amp)), orig_amp, '-', label=f"Original")
        axes[0].plot(np.arange(len(recon_amp)), recon_amp, '-', label=f"Reconstructed")
        axes[1].plot(np.arange(len(orig_freq)), orig_freq, '-', label=f"Original")
        axes[1].plot(np.arange(len(recon_freq)), recon_freq, '-', label=f"Reconstructed")

    for i, axlabel in enumerate(['Amplitude', 'Frequency']):
        axes[i].set_xlabel('Sample length', fontsize=12)
        axes[i].set_ylabel(axlabel, fontsize=12)
        axes[i].xaxis.set_minor_locator(tck.AutoMinorLocator())
        axes[i].yaxis.set_minor_locator(tck.AutoMinorLocator())
        axes[i].tick_params(which='both', direction='in', top=True, right=True)
    axes[1].legend(fontsize=8, loc='upper left')

    # Place suptitle closer to the top of the figure, not too far away
    plt.suptitle(
        f'$m_1$={float(label[0]):.2f}, $m_2$={float(label[1]):.2f}, '
        f'$\\chi_1(z)$={float(label[2]):.2f}, $\\chi_2(z)$={float(label[3]):.2f}',
        fontsize=12,
        y=0.95  # Move title closer to the top edge (default is 0.99)
    )

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
        axes[i].set_xlabel('Sample length', fontsize=15)
        axes[i].xaxis.set_minor_locator(tck.AutoMinorLocator())
        axes[i].yaxis.set_minor_locator(tck.AutoMinorLocator())
        axes[i].tick_params(which='both', direction='in', top=True, right=True)
        axes[i].tick_params(axis='both', labelsize=12)
    axes[0].set_ylabel('$h_{+}$', fontsize=15)
    axes[2].set_ylabel('$h_{\\times}$', fontsize=15)
    axes[0].legend(fontsize=8, loc='upper left')
    axes[2].legend(fontsize=8, loc='upper left')

    # Place suptitle closer to the top of the figure, not too far away
    plt.suptitle(
        f'$m_1$={float(label[0]):.2f}, $m_2$={float(label[1]):.2f}, '
        f'$\\chi_1(z)$={float(label[2]):.2f}, $\\chi_2(z)$={float(label[3]):.2f}',
        fontsize=15,
        y=0.97  # Move title closer to the top edge (default is 0.99)
    )

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
    x = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
    reconst = reconst.cpu().numpy() if isinstance(reconst, torch.Tensor) else reconst
    labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    keys = keys.cpu().numpy() if isinstance(keys, torch.Tensor) else keys

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

        # -- Remove first dummy element from frequency array!
        orig_freq = orig_freq[1:]
        recon_freq = recon_freq[1:]
        
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

        try:
            assert len(orig_amp) == len(recon_amp), f"Original and reconstructed amplitude arrays must have the same length. Got {len(orig_amp)} and {len(recon_amp)}"
            assert len(orig_freq) == len(recon_freq), f"Original and reconstructed frequency arrays must have the same length. Got {len(orig_freq)} and {len(recon_freq)}"
        except AssertionError as e:
            # logging.error(f"Length mismatch between original and reconstructed data for sample {i}: {e}")
            logging.debug(f"Original amplitude length: {len(orig_amp)}, Reconstructed amplitude length: {len(recon_amp)}")
            # logging.warning("Removing first element from original amplitude and frequency arrays to match reconstructed data length.")
            orig_amp = orig_amp[1:]
            orig_freq = orig_freq[1:]
            logging.debug(f"After removing first element, Original amplitude length: {len(orig_amp)}, Reconstructed amplitude length: {len(recon_amp)}")
        
        # Calculate mismatch
        mismatch_amp[i] = calculate_mismatch(orig_amp, recon_amp)
        mismatch_freq[i] = calculate_mismatch(orig_freq, recon_freq)
        logging.info(f"Mismatch for Amplitude: {mismatch_amp[i]}, Frequency: {mismatch_freq[i]}")

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
            logging.info(f"Mismatch plot saved to {savedir+savename}.png")
            plt.close()
    return mismatch_amp, mismatch_freq, chirpmasses, totalmasses, massratios



def plot_polarization_mismatch(x, reconst, labels, keys, phases, strains, attr,
                               reshape2orig=False,
                               savedir='../results/', nobatchwiseplot=False,
                               num_saved_overplots=0, generating=False):
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
    strains : torch.Tensor
        Strain information associated with the original data.
    attrs : dict
        Additional attributes associated with the data.
        
    Returns:
    --------
    None

    Outputs:
    -------
    Displays a plot of the polarization mismatch values.
    """
    x = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
    reconst = reconst.cpu().numpy() if isinstance(reconst, torch.Tensor) else reconst
    labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    keys = keys.cpu().numpy() if isinstance(keys, torch.Tensor) else keys
    phases = phases.cpu().numpy() if isinstance(phases, torch.Tensor) else phases
    strains = strains.cpu().numpy() if isinstance(strains, torch.Tensor) else strains
    logging.debug(f"x shape: {x.shape}, reconst shape: {reconst.shape}, phases shape: {phases.shape}, strains shape: {strains.shape}")

    chirpmasses = np.zeros((labels.shape[0], 1))  # Store chirp masses for each sample
    totalmasses = np.zeros((labels.shape[0], 1))  # Store total masses for each sample
    massratios = np.zeros((labels.shape[0], 1))  # Store mass ratios for each sample
    mismatch_hplus = np.zeros((x.shape[0], 1))
    mismatch_hcross = np.zeros((x.shape[0], 1))

    chieffs = np.zeros((labels.shape[0], 1))  # Store chi_eff for each sample
    if labels.shape[1] == 4:
        # labels are [m1, m2, spin1z, spin2z]
        for i in range(labels.shape[0]):
            m1, m2 = labels[i][0], labels[i][1]
            chi1, chi2 = labels[i][2], labels[i][3]
            chi_eff = (m1 * chi1 + m2 * chi2) / (m1 + m2)
            chieffs[i] = chi_eff

    # iterate over all the waveforms in one batch
    for i in range(x.shape[0]):
        logging.debug(f"Processing sample {i}")
        if reshape2orig:
            # Reshape to original data shape
            orig_data = x[i].reshape([2,PRESET_ARRAY_SIZE])
            reconst = reconst.reshape([2,PRESET_ARRAY_SIZE])
            phase = phases[i].reshape([1,PRESET_ARRAY_SIZE])
        else:
            # Use the original shape of the data
            orig_data = x[i].reshape([2,x.shape[2]])
            recon_data = reconst[i].reshape([2,reconst.shape[2]])
            logging.debug(f"orig_data shape: {orig_data.shape}, recon_data shape: {recon_data.shape}")
            phase = phases[i].reshape([phases.shape[1]])          
            logging.debug(f'phase shape: {phase.shape}')

        # print(type(attr), attr.keys(), type(attr['delta_t'][i]), attr['f_lower'][i])
        delta_t = attr['delta_t'][i]
        f_lower = attr['f_lower'][i]

        orig_amp, orig_freq = orig_data[0], orig_data[1]
        recon_amp, recon_freq = recon_data[0], recon_data[1]

        # -- remove first dummy element from frequency array!
        orig_freq = orig_freq[1:]
        recon_freq = recon_freq[1:]
        
        # Obtain the keys for normalization
        key = keys[i].reshape([2,2])
        amp_mean, amp_std = key[0][0], key[0][1]
        freq_mean, freq_std = key[1][0], key[1][1]

        # De-normalize the original data!
        if not generating:
            # NOTE: If not generating samples, then data is loaded via CustomDataset
            # and needs de-normalization.
            orig_amp = (orig_amp * amp_std) + amp_mean
            orig_freq = (orig_freq * freq_std) + freq_mean
            logging.debug(f"original amp shape: {orig_amp.shape}, freq shape: {orig_freq.shape}")
        # De-normalize the reconstructed data!
        recon_amp = (recon_amp * amp_std) + amp_mean
        recon_freq = (recon_freq * freq_std) + freq_mean
        logging.debug(f"reconstructed amp shape: {recon_amp.shape}, freq shape: {recon_freq.shape}")

        # Rescale the reconstructed amplitude to match the original amplitude's maximum value, 
        # to avoid mismatch due to amplitude scaling differences.
        # recon_amp = recon_amp / 10**20

        # # check length of phase array
        # # NOTE: This happens because of the f-cutoff datacase!
        # if len(phase) != orig_amp.shape[0]:
        #     logging.warning(f"Phase shape {phase.shape} does not match input shape {orig_amp.shape}. Adjusting phase.")
        #     # Adjust phase to match orig_amp length
        #     if len(phase) > orig_amp.shape[0]:
        #         phase = phase[1:]
        #     assert len(phase) == orig_amp.shape[0]

        # Combine original Amp/Freq to hplus/hcross
        # hp_orig = orig_amp * np.cos(phase)  # this is original phase
        # hc_orig = orig_amp * np.sin(phase)
        hp_hdf = strains[i][0]
        hc_hdf = strains[i][1]
        # print(max(hp_orig), max(hc_orig))
        logging.debug(f"Original hplus shape: {hp_hdf.shape}, hcross shape: {hc_hdf.shape}")

        # -- NOTE: what we instead do now to remove the errors and
        # -- and the possibility of retraining is to have the darn nice,
        # -- same length amplitude and freq arrays for both the original
        # -- and the model outputs, by using the shortened amplitude arrays
        # -- and repeating the first element in them to make them the same
        # -- length as the frequency arrays, and then use the `_phase_from_freq_intervals`.
        phase_hdf = np.unwrap(np.arctan2(hp_hdf, hc_hdf))
        hp_orig, hc_orig = polarizations_from_ampfreq(orig_amp, orig_freq, theta0=phase_hdf[0])

        # Calculate hplus/hcross for reconstructed data
        hp_recon, hc_recon = polarizations_from_ampfreq(recon_amp, recon_freq, theta0=phase_hdf[0])
        if num_saved_overplots is not None:
            if num_saved_overplots <= 10:
                plot_hphc_overplot(hp_orig, hc_orig, hp_recon, hc_recon, label=labels[i],
                                savename=savedir+'overplot-hphc')
                num_saved_overplots += 1

        try:
            assert hp_orig.shape == hp_recon.shape, f"Original and reconstructed hplus waveforms must have the same shape. Got {hp_orig.shape} and {hp_recon.shape}"
            assert hc_orig.shape == hc_recon.shape, f"Original and reconstructed hcross waveforms must have the same shape. Got {hc_orig.shape} and {hc_recon.shape}"
        except AssertionError as e:
            # logging.error(f"Shape mismatch between original and reconstructed hplus/hcross for sample {i}: {e}")
            logging.debug(f"Original hplus shape: {hp_orig.shape}, Reconstructed hplus shape: {hp_recon.shape}")
            # logging.warning("Removing first element of the orig array to make them equal length")
            hp_orig = hp_orig[1:]
            hc_orig = hc_orig[1:]
            

        # Calculate mismatch for hplus and hcross
        # NOTE: Can use `hp_hdf` and `hc_hdf` for this!
        mismatch_hplus[i] = calc_polarization_mismatch(hp_orig, hp_recon, delta_t=delta_t, f_lower=f_lower)
        mismatch_hcross[i] = calc_polarization_mismatch(hc_orig, hc_recon, delta_t=delta_t, f_lower=f_lower)
        logging.info(f"Mismatch for hplus: {mismatch_hplus[i]}, hcross: {mismatch_hcross[i]}")

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
            # ax.plot(massarr, mismatch_hplus, '.', markeredgewidth=0.75, alpha=0.75)
            # ax.plot(massarr, mismatch_hcross, '.', markeredgewidth=0.75, alpha=0.75)
            # Set legend for each sample with its label
            for i in range(labels.shape[0]):
                ax.plot(massarr[i], mismatch_hplus[i], random.choice(markers),
                        label=f'{labels[i][0]:.2f}, {labels[i][1]:.2f}, {labels[i][2]:.2f}, {labels[i][3]:.2f}', markeredgewidth=0.75, alpha=0.75)
                # ax.plot(massarr[i], mismatch_hcross[i], 'x', markeredgewidth=0.75, alpha=0.75)
            ax.legend(fontsize=5, loc='upper left', title='$m_1$, $m_2$, $\\chi_1(z)$, $\\chi_2(z)$')
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
    return (mismatch_hplus, mismatch_hcross, chirpmasses, totalmasses, massratios, chieffs, num_saved_overplots)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Non-Eccentric GW Generator')

    parser.add_argument('--nsamples', action='store', default=1000, type=int,
                            help='default=%(default)s')
    parser.add_argument('--approximant', action='store', default='SEOBNRv4', )
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
    parser.add_argument('--test-uq-iter', action='store_true', default=False,
                        help='whether to test uncertainty quantification iteratively for Nwaves?')
    parser.add_argument('--time-complexity', action='store_true', default=False,
                        help='whether to test time complexity?')
    parser.add_argument('--time-compare', action='store_true', default=False,
                        help='whether to compare time complexity with standard waveform generation?')
    parser.add_argument('--test-mm-compare', action='store_true', default=False,
                        help='whether to compare mismatch with standard waveform generation?')
    parser.add_argument('--generate', action='store_true', default=False,
                            help='whether to generate samples from trained model?')

    parser.add_argument('--model', action='store', default=None,
                        help='name of a pre-trained model.')
    parser.add_argument('--modeltype', action='store', default='cvae', choices=['cvae', 'cae', 'flexcvae'],
                        help='type of model to use, e.g., cvae or cae (default=%(default)s)')
    parser.add_argument('--usemmloss', action='store_true', default=False,
                        help='whether to use mismatch loss during training (default=%(default)s)')
    parser.add_argument('--use-base-model-config', action='store_true', default=False,
                        help='Whether to use the base model config for training? (default=%(default)s)')
    
    parser.add_argument('--today', action='store', default=None,
                        help='Date of the model we are currently using, in YYYYMMDD. \
                            Results will be saved to this folder. (default=%(default)s)')
    parser.add_argument('--fname', action='store', default=None,
                        help='Dummy filename argument. (default=%(default)s)')
    parser.add_argument('--savedir', action='store', default=None,
                        help='Directory to save results. (default=%(default)s)')
    
    parser.add_argument('--noshow', action='store_true', default=False,
                            help='Do not show output Plot !')
    parser.add_argument('--nosave', action='store_true', default=False,
                            help='Do not save output files and plots!')
    parser.add_argument('--dummy', action='store_true', default=False,
                            help='Whether to use dummy data for testing the code. (default=%(default)s')
    parser.add_argument('--fix-random-seed', action='store_true', default=False,
                            help='Whether to fix random seed for reproducibility? (default=%(default)s)')
    parser.add_argument('--random-seed', action='store', default=42, type=int,
                            help='Random seed value (default=%(default)s)')

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
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    logfname = f'training_{now}.log' if not args.test and not args.generate else f'testing_{today}.log'
    log_file = os.path.join(log_dir, logfname)
    logging.basicConfig(
        format='%(levelname)s | %(asctime)s: %(message)s',
        level=log_level,
        datefmt='%y-%m-%d %H:%M:%S',
        force=True,
        handlers=[
            logging.StreamHandler(),  # Log to console
            logging.FileHandler(log_file) # Log to file
        ]
    )

    # Set FileHandler to always be at least INFO level
    for handler in logging.root.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setLevel(max(handler.level, logging.INFO))
    
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
        # try:
        if args.test_uq:
            Test(args).test_uq(batch_size=args.batch_size, Nruns=5000, 
                               plot_hist=True, plotonlyone=True,
                               fontsize=15, labelsize=13)
        elif args.test_uq_iter:
            Test(args).test_uq_iter(Nwaves=1000, Nruns=100,
                                    fontsize=15, labelsize=13)
            Test(args).test_uq_iter(Nwaves=1000, Nruns=1000,
                                    fontsize=15, labelsize=13)
            Test(args).test_uq_iter(Nwaves=5000, Nruns=100,
                                    fontsize=15, labelsize=13)
        elif args.time_complexity:
            # for n in [100, 500, 1000]:
            Test(args).test_timecomplexity(num_start=1, num=100)
        elif args.time_compare:
            if args.fname is not None:
                Test(args).plot_time_complexity_compare(fname=args.fname)
            else:
                Test(args).test_timecomplexity_compare(iters=100)
        elif args.test_mm_compare:
            Test(args).test_mismatch_compare()
        else:
            Test(args).test()
        # except RuntimeError as e:
        #     logging.error(f"Error occurred during testing (perhaps try `--fcutoff`): {e}")
    elif args.generate:
        labels = np.array([[50.0, 15.0, 0.0, 0.0],
                           [30.0, 5.0, 0.0, 0.0],
                           [50.0, 15.0, 0.5, 0.5],
                           [30.0, 5.0, 0.5, 0.5],
                           [50.0, 15.0, -0.5, -0.5],
                           [30.0, 5.0, -0.5, -0.5],
                           [50.0, 15.0, 0.9, 0.9],
                           [30.0, 5.0, 0.9, 0.9]])
        args.nsamples = labels.shape[0]
        Test(args).generate(labels=labels)
    else:
        # try:
        #     # Always reads data from HDF file now!!
        #     train(args)
        # except RuntimeError as e:
        #     logging.error(f"Error occurred during training (perhaps, ya forgot to put `--fcutoff`): {e}")
        train(args)
