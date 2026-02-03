"""
Use Optuna to optimize hyperparameters for training the CVAE model on gravitational waveforms.
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

# from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary

import optuna

from datetime import datetime

import sys
sys.path.append('~Dropbox/plotutils-work/plotutils/')
from plotutils import putils

from datacvae import CustomDataset, CustomDataLoader
from datacvae import PRESET_ARRAY_SIZE, SAMPLE_RATE, DELTA_T, f_lower, sample_len
from cvae import CVAE




markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'X', 'h', '1', '2', '3', '4', '8']



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
    elif args.aligned:
        num_classes = 4  # m1, m2, spin1z, spin2z
        trainhdf += '-100000-fcutoff-uniform-aligned'
        # trainhdf += '-4e5-fcutoff-uniform-aligned'
        validhdf += '-100000-fcutoff-uniform-aligned'
    
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
    train_set = CustomDataset(forwhat='train', approximant=args.approximant,
                            convert=args.convert, hdf_fname=trainhdf, train_device=args.device)
    logging.info(f'Reading validation data from {validhdf}.hdf')
    valid_set = CustomDataset(forwhat='valid', approximant=args.approximant,
                            convert=args.convert, hdf_fname=validhdf, train_device=args.device)
    # logging.info(f"Train set size: {len(train_set)}")
    # logging.info(f"Validation set size: {len(valid_set)}")

    # try:
    #     training_loader = CustomDataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    #     validation_loader = CustomDataLoader(valid_set, batch_size=args.batch_size, shuffle=True)
    # except ValueError:
    #     training_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    #     validation_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True)
    training_loader = CustomDataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    validation_loader = CustomDataLoader(valid_set, batch_size=args.batch_size, shuffle=True)
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
        for x, target, labels, keys in tqdm(training_loader, total=len(training_loader),
                                    desc='batch'):
            """
            `x` is [freq, amp], `labels` is [m1,m2] etc. and
            `keys` is [[amp-mean,amp-var],[freq-mean,freq-var]]
            """
            x, target, labels, keys = x.to(args.device), target.to(args.device), labels.to(args.device), keys.to(args.device)
            optimizer.zero_grad()
            x_recon, zvars = model(x, labels, keys)
            
            # TODO: have it such that the training target are unnormalized waveforms!
            # loss, reconloss, klloss = model.loss_function(x, x_recon, zvars)
            loss, reconloss, klloss = model.loss_function(target, x_recon, zvars)
            loss.backward()
            train_rloss.append(loss.item())
            netreconloss.append(reconloss.item())
            netklloss.append(klloss.item())
        train_loss.append(loss.item())

        model.eval()
        with torch.no_grad():  # Disable gradient computation for validation
            # just reconstruct target and calculate diff
            for vx, target, vlabels, vkeys in tqdm(validation_loader, desc='val-batch'):
                vx, target, vlabels, vkeys = vx.to(args.device), target.to(args.device), vlabels.to(args.device), vkeys.to(args.device)
                vx_recon, vzvars = model(vx, vlabels, vkeys)
                vloss, _reconloss, _klloss = model.loss_function(target, vx_recon, vzvars)
                valid_rloss.append(vloss.item())
            valid_loss.append(vloss.item())
        tqdm.write(f'Epoch {epoch+1} : train loss {loss.item()} & valid loss {vloss.item()}')
        
        optimizer.step()  # Do optimization step after validation
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
    parser.add_argument('--fcutoff', action='store_true', default=False,
                            help='use the data which has equal duration samples with \
                                variable frequency cutoff (default=%(default)s)')
    parser.add_argument('--aligned', action='store_true', default=False,
                            help='use the aligned-spin data uniformly sampled in (m1,m2,spin1z,spin2z) space.\
                                (default=%(default)s)')

    parser.add_argument('--today', action='store', default=None,
                        help='Date of the model we are currently using, in YYYYMMDD. \
                            Results will be saved to this folder. (default=%(default)')
    parser.add_argument('--fname', action='store', default=None,
                        help='Dummy filename argument. (default=%(default)s)')
    parser.add_argument('--savedir', action='store', default=None,
                        help='Directory to save results. (default=%(default)s)')
    
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

    train(args)
