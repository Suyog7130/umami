"""
Uses Optuna with `multicvae` for optimization of the
hyper-parameters and number of layers etc.
"""

import os

# -- Allow OpenMP to use all available CPU cores, based on the config file.
os.environ["KMP_AFFINITY"] = "disabled"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


import gc
import h5py
import json
import argparse
import pandas as pd
import datetime
import joblib

from tqdm import tqdm

import optuna

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np

from datacvae import CustomDataset, CustomDataLoader
# from multicvae import TwoC2E1D

from flexcvae import FlexTwoC2E1D, FlexCAE, FlexCAEPhase
from cvae import CVAE


from utils.gwutils import (
    calculate_cosine_distance,
    polarizations_from_amp_phase,
    calc_polarization_mismatch,
    calc_chirp_mass,
    calc_chieff,
    calc_time_array
)
from utils.io import ensure_dir
from utils.plotting import plot_twopanel

import logging
from utils.generic import init_logging, init_verbosity_args
logger = logging.getLogger(__name__)


PROJECT_DIR = 'v0p1'


TODAY = datetime.date.today().strftime("%Y%m%d")
TIME = datetime.datetime.now().strftime("%H%M%S")
NOW = TODAY + '-' + TIME

DATAFRAC = 0.1  # Fraction of training data to use for quick training during Optuna optimization
EPOCHS = 10
BATCH_SIZE = 64
PRESET_ARRAY_SIZE = 8191
APPROXIMANT = 'SEOBNRv4'
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    PRECISION = 'float32'  # Can also use double precision `float64` for CUDA if available
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    PRECISION = 'float32'  # Use float32 for MPS since it does not support float64 well
else:
    DEVICE = torch.device("cpu")
    PRECISION = 'float32'
# print(f"Using device: {DEVICE}, with precision: {PRECISION}")

BASE_MODEL_CONFIG = {
    'latent_dim_x': 16,
    'latent_dim_key': 4,
    'activation': 'gelu',
    'train_device': DEVICE,
    'model_precision': PRECISION,
    'modeltype': 'flexcvae',
    'loss_func_type': None,
    'target': None, #'amp_phase',  # default target is normed amp-freq, but can be set to 'logamp_phase' for log-amp and phase target
}

datadir = "../data/"
train_hdf = datadir + 'SEOBNRv4-train-100000-fcutoff-uniform-aligned-regen'
val_hdf = datadir + "SEOBNRv4-val-100000-fcutoff-uniform-aligned-regen"
test_hdf = datadir + "SEOBNRv4-test-100000-fcutoff-uniform-aligned-regen"

# -- get mean and std of labels for normalization
params_fname = '../data/params-' + APPROXIMANT + '-train-100000-fcutoff-uniform-aligned-regen-4vals.csv'
params_df = pd.read_csv(params_fname, index_col=0, sep=',')
params_mean = params_df.mean().values
params_std = params_df.std().values
logger.info(f"Labels mean: {params_mean}")
logger.info(f"Labels std: {params_std}")
params_mean = torch.tensor(params_mean, dtype=getattr(torch, PRECISION))
params_std = torch.tensor(params_std, dtype=getattr(torch, PRECISION))


def set_dataloaders(batch_size=BATCH_SIZE, target=BASE_MODEL_CONFIG['target']):
    """
    Sets up the dataloaders for training and validation datasets.
    Arguments:
        batch_size: Batch size for the dataloaders (default BATCH_SIZE)
        target: Target for the model (default BASE_MODEL_CONFIG['target'])
    Returns:
        train_loader: DataLoader for the training dataset
        val_loader: DataLoader for the validation dataset
    """
    logger.info(f'Reading training data from {train_hdf}.hdf')
    train_set = CustomDataset(forwhat='train', approximant=APPROXIMANT, returnattr=True,
                            hdf_fname=train_hdf, train_device=DEVICE, precision=PRECISION,
                            target=target)
    logger.info(f'Reading validation data from {val_hdf}.hdf')
    valid_set = CustomDataset(forwhat='valid', approximant=APPROXIMANT, returnattr=True,
                            hdf_fname=val_hdf, train_device=DEVICE, precision=PRECISION, 
                            target=target)
    # NOTE: 'pin_memory' doesn't work if we already passed the data to GPU inside the CustomDataset!
    train_loader = CustomDataLoader(train_set, batch_size=batch_size, shuffle=True,
                                    num_workers=8, pin_memory=False)
    val_loader = CustomDataLoader(valid_set, batch_size=batch_size, shuffle=False, 
                                  num_workers=8, pin_memory=False)
    logger.info(f"Training dataset size: {len(train_set)}, Validation dataset size: {len(valid_set)}")
    return train_loader, val_loader


def save_data_from_dataset(savedir='../data/',
                           input_type='amp_phase', target_type='amp_phase',
                           input_normalized=True, target_normalized=False):
    """
    Save input and target waveform data the datasets
    """
    assert input_type == target_type, "Both input and target types should be same, normalization can differ."
    label = f'{APPROXIMANT}-{target_type}'
    label += '-normIN' if input_normalized else '-unnormIN'
    label += '-normOUT' if target_normalized else '-unnormOUT'
    data_dir = '/Users/suyoggarg/Desktop/'
    # train_hdf_path = data_dir + 'waveforms-SEOBNRv4-aligned-fcutoff-1sec-8192Hz-train-70k-seed42-params_002-20260430-225157.hdf'
    # val_hdf_path = data_dir + 'waveforms-SEOBNRv4-aligned-fcutoff-1sec-8192Hz-val-10k-seed42-params_002-20260430-225157.hdf'
    # test_hdf_path = data_dir + 'waveforms-SEOBNRv4-aligned-fcutoff-1sec-8192Hz-test-20k-seed42-params_002-20260430-225157.hdf'
    train_hdf_path = train_hdf
    val_hdf_path = val_hdf
    test_hdf_path = test_hdf
    train_set = CustomDataset(approximant=APPROXIMANT,
                            hdf_fname=train_hdf_path, 
                            train_device=DEVICE, precision=PRECISION,
                            input_type=input_type, target_type=target_type,
                            input_normalized=input_normalized, target_normalized=target_normalized,
                            return_attributes=True, return_phases=True, return_sample_indices=False)
    valid_set = CustomDataset(approximant=APPROXIMANT,
                            hdf_fname=val_hdf_path, 
                            train_device=DEVICE, precision=PRECISION,
                            input_type=input_type, target_type=target_type,
                            input_normalized=input_normalized, target_normalized=target_normalized,
                            return_attributes=True, return_phases=True, return_sample_indices=False)
    test_set = CustomDataset(approximant=APPROXIMANT,
                            hdf_fname=test_hdf_path, 
                            train_device=DEVICE, precision=PRECISION,
                            input_type=input_type, target_type=target_type,
                            input_normalized=input_normalized, target_normalized=target_normalized,
                            return_attributes=True, return_phases=True, return_sample_indices=False)
    train_set.save_to_input_file(savename=savedir+f'{label}-train.hdf')
    valid_set.save_to_input_file(savename=savedir+f'{label}-val.hdf')
    test_set.save_to_input_file(savename=savedir+f'{label}-test.hdf')



class WaveformDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset for loading waveform data from HDF5 files.
    This is a simplified version that only loads the input waveforms and targets,
    without labels, keys, strains, or additional attributes.
    """
    def __init__(self, hdf_fname, 
                 target_type: {'amp_freq', 'logamp_freq', 'amp_phase', 'logamp_phase'} = 'amp_phase',
                 input_normalized=True, target_normalized=False,
                 params_mean=None, params_std=None,
                 device=DEVICE, precision=PRECISION,
                 return_indices=False):
        super(WaveformDataset, self).__init__()
        self.hdf_fname = hdf_fname
        self.train_device = device
        self.precision = precision
        self.return_indices = return_indices

        self.target_type = self.input_type = target_type
        self.input_normalized = input_normalized
        self.target_normalized = target_normalized
        self._set_input_target_names()

        # self.params_mean = params_mean
        # self.params_std = params_std
        # logger.info(f'Parameters mean: {self.params_mean}, Parameters std: {self.params_std}')
        # assert self.params_mean is None or self.params_mean.shape == (4,), f"Expected params_mean to be of shape (4,), but got {self.params_mean.shape}"
        # assert self.params_std is None or self.params_std.shape == (4,), f"Expected params_std to be of shape (4,), but got {self.params_std.shape}"

        # # NOTE: `init_hdf` inside `__getitem__` call, to allow spawning multiple workers for dataloading.
        # self.init_hdf()  # initialize the HDF file for reading the data in the `__getitem__` method

        # temporarily open file to get length
        with h5py.File(self.hdf_fname, 'r') as f:
            # Store the length as a simple integer (safe to pickle!)
            self.length = len(f.keys()) 
            self.group_names = list(f.keys())

    def _set_input_target_names(self):
        # -- labels for input and target data for different kinds of targets.
        name_dict = {'amp_freq': ['amp', 'freq'],
                    'logamp_freq': ['logamp', 'freq'],
                    'amp_phase': ['amp', 'phase'],
                    'logamp_phase': ['logamp', 'phase']}
        self.inputnames, self.targetnames = name_dict[self.input_type], name_dict[self.target_type]
        if self.input_normalized:
            self.inputnames = ['normed_' + name for name in self.inputnames]
        else:
            self.inputnames = ['unnormed_' + name for name in self.inputnames]
        if self.target_normalized:
            self.targetnames = ['normed_' + name for name in self.targetnames]
        else:
            self.targetnames = ['unnormed_' + name for name in self.targetnames]

    def __len__(self):
        return self.length

    def init_hdf(self):
        # -- check if the HDF file exists, if not, create an empty HDF file with the same name, so that we can write to it later on in the `get_calibrator_input` function without having to worry about file not found errors.
        hdf_fname = self.hdf_fname + '.hdf' if not self.hdf_fname.endswith('.hdf') else self.hdf_fname
        data_path = f'../data/{hdf_fname}' if not hdf_fname.startswith('../data/') else hdf_fname

        if not os.path.exists(data_path):
            logger.error(f"HDF file {data_path} does not exist. Please run the `save_data_from_dataset` function to create the HDF file with the input and target data before initializing the dataset.")
        else:
            logger.info(f"HDF file {data_path} already exists. Will read from it or append to it when generating calibrator input and target residuals.")

        # open the HDF file for reading in the dataset initialization, so that we can read from it in the `__getitem__` method without having to open and close the file every time, which is inefficient. We will keep this file open for the lifetime of the dataset, and close it when the dataset is deleted.
        self.data_file = h5py.File(data_path, 'r')
        logger.info(f"Opened HDF file {data_path} for reading or writing data.")

    def close_hdf(self):
        # close the HDF file when the dataset is deleted, to free up resources
        if hasattr(self, 'data_file') and self.data_file is not None:
            self.data_file.close()
            logger.info(f"Closed HDF file {self.hdf_fname}")

    # def __del__(self):
    #     self.close_hdf()


    def collate_fn(self, batch):
        """ 
        Custom collate function to handle the batch data.
        To account for some `bool` values in the `feat_dict` that are not tensors, 
        we will separate the `feat_dict` from the rest of the batch data and handle it separately.

        NOTE: It is assumed that the `feat_dict` will always the last in the batch tuple!

        NOTE: We move the tensors to train device inside the training function, not here!
        """
        tag_batches = []
        feat_dict_batch = {}
        logger.debug(f'Batch size: {len(batch)}')

        max_tags = max(len(sample) - 1 for sample in batch)  # Exclude the feature dict

        # Initialize lists for each tag dynamically
        for _ in range(max_tags):
            tag_batches.append([])

        for sample in batch:
            *tags, feat_dict = sample
            logger.debug(f'Number of tags: {len(tags)}')
            for key, value in feat_dict.items():
                if key not in feat_dict_batch:
                    feat_dict_batch[key] = []
                if isinstance(value, bool):
                    # Convert bool to int for storage in HDF5
                    value = int(value)
                feat_dict_batch[key].append(value)

            # Append tags to their respective lists
            # NOTE: tags are already tensors!
            for i, tag in enumerate(tags):
                if not isinstance(tag, torch.Tensor):
                    tag = torch.tensor(tag, dtype=getattr(torch, self.precision))
                tag_batches[i].append(tag)

        # Convert lists of tags to tensors
        for i in range(len(tag_batches)):
            tag_batches[i] = torch.stack(tag_batches[i]).to(dtype=getattr(torch, self.precision))

        return (*tag_batches, feat_dict_batch)

    def read_data_from_hdf(self, idx):
        # -- read the input waveform and target residual from the HDF file for the given index, and return them as tensors
        if not hasattr(self, 'data_file') or self.data_file is None:
            self.init_hdf()
        grp = f'sample{idx}'
        # logger.debug(f"Reading data for index {idx} from group {grp} in HDF file {self.hdf_fname}.")
        if grp not in self.data_file:
            logger.error(f"Group {grp} not found in HDF file {self.hdf_fname}. Cannot read data for index {idx}.")
            raise KeyError(f"Group {grp} not found in HDF file {self.hdf_fname}.")
        data = self.data_file[grp]
        # logger.debug(f"Available datasets in group {grp}: {list(data.keys())}")
        input_one = data[f'input_{self.inputnames[0]}'][:]
        input_two = data[f'input_{self.inputnames[1]}'][:]
        target_one = data[f'target_{self.targetnames[0]}'][:]
        target_two = data[f'target_{self.targetnames[1]}'][:]
        input = torch.stack([torch.tensor(input_one, dtype=getattr(torch, self.precision)),
                            torch.tensor(input_two, dtype=getattr(torch, self.precision))], dim=0)
        target = torch.stack([torch.tensor(target_one, dtype=getattr(torch, self.precision)),
                             torch.tensor(target_two, dtype=getattr(torch, self.precision))], dim=0)
        labels = torch.tensor(data['labels'][:], dtype=getattr(torch, self.precision))
        keys = torch.tensor(data['keys'][:], dtype=getattr(torch, self.precision))
        strains = torch.tensor(data['strains'][:], dtype=getattr(torch, self.precision))
        attr = dict(data.attrs)
        logger.debug(f"Available attributes in group {grp}: {list(attr.keys())}")
        if self.return_indices:
            return input, target, labels, keys, strains, np.array(idx), attr
        return input, target, labels, keys, strains, attr

    def __getitem__(self, idx):
        return self.read_data_from_hdf(idx)
    
    def get_data_by_groupname(self, grpname):
        # -- read the input waveform and target residual from the HDF file for the given group name, and return them as tensors
        if not hasattr(self, 'data_file') or self.data_file is None:
            self.init_hdf()
        if grpname not in self.group_names:
            logger.error(f"Group {grpname} not found in HDF file {self.hdf_fname}. Cannot read data for group name {grpname}.")
            raise KeyError(f"Group {grpname} not found in HDF file {self.hdf_fname}.")
        idx = grpname.replace('sample', '')  # extract index from group name
        return self.read_data_from_hdf(idx)


class WaveformDataLoader(torch.utils.data.DataLoader):
    """
    A custom dataloader for the `WaveformDataset`, which inherits from `torch.utils.data.DataLoader`!
    It allows for shuffling and batching of the waveform data.
    """
    def __init__(self, dataset, batch_size=32, shuffle=True, **kwargs):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                         collate_fn=dataset.collate_fn, **kwargs)


def set_waveform_dataloaders(batch_size=BATCH_SIZE,
                             num_workers=8,
                             return_test_loader=False,
                             target_type: {'amp_freq', 'logamp_freq', 'amp_phase', 'logamp_phase'} = 'amp_phase',
                             input_normalized=True, target_normalized=False,
                             return_indices=False):
    """
    Sets up the dataloaders for training and validation datasets for preprocessed input
    and target data. We will autoconstruct the filenames from the passed arguements.

    Arguments
    ---------
    batch_size: int
        Batch size for the dataloaders (default BATCH_SIZE)
    return_test_loader: bool
        Whether to return the test dataloader along with the train and validation dataloaders (default False)
    target_type: str
        Target type for the model, one of 'amp_freq', 'logamp_freq', 'amp_phase', 'logamp_phase' (default 'amp_phase')
    input_normalized: bool
        Whether the input data is normalized (default True)
    target_normalized: bool
        Whether the target data is normalized (default False)
    """
    label = f'{APPROXIMANT}-{target_type}'
    label += '-normIN' if input_normalized else '-unnormIN'
    label += '-normOUT' if target_normalized else '-unnormOUT'
    train_hdf_path = f'../data/{label}-train.hdf'
    val_hdf_path = f'../data/{label}-val.hdf'
    test_hdf_path = f'../data/{label}-test.hdf'
    logger.info(f"Setting up dataloaders with train HDF: {train_hdf_path}, val HDF: {val_hdf_path}, test HDF: {test_hdf_path}")
    if return_test_loader:
        test_set = WaveformDataset(hdf_fname=test_hdf_path, target_type=target_type,
                                input_normalized=input_normalized, target_normalized=target_normalized,
                                params_mean=params_mean, params_std=params_std,
                                train_device=DEVICE, precision=PRECISION, return_indices=return_indices)
        test_loader = WaveformDataLoader(test_set, batch_size=batch_size, shuffle=False,
                                        num_workers=num_workers, pin_memory=True,
                                        drop_last=False)
        logger.info(f"Test DataLoader set up with batch size {batch_size} and {num_workers} workers.")
        return test_loader
    train_set = WaveformDataset(hdf_fname=train_hdf_path, target_type=target_type,
                                input_normalized=input_normalized, target_normalized=target_normalized,
                                params_mean=params_mean, params_std=params_std,
                                train_device=DEVICE, precision=PRECISION, return_indices=return_indices)
    val_set = WaveformDataset(hdf_fname=val_hdf_path, target_type=target_type,
                              input_normalized=input_normalized, target_normalized=target_normalized,
                              params_mean=params_mean, params_std=params_std,
                              train_device=DEVICE, precision=PRECISION, return_indices=return_indices)
    logger.info(f"Initialized WaveformDataset for train and val sets.")
    # NOTE: `drop_last=True` in train_loader, ensures that all batches have equal size, and thus makes `torch.compile` make the training faster, once the model has been precompiled. For validation and test loaders, we can keep `drop_last=False`, since we want to evaluate on all samples.
    train_loader = WaveformDataLoader(train_set, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers, pin_memory=True,
                                       drop_last=True)
    val_loader = WaveformDataLoader(val_set, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers, pin_memory=True,
                                      drop_last=False)
    logger.info(f"DataLoaders set up with batch size {batch_size} and {num_workers} workers.")
    return train_loader, val_loader



def training(model: {FlexTwoC2E1D, FlexCAE, FlexCAEPhase}, 
             train_loader=None, val_loader=None,
             epochs: int = 5, 
             init_lr: float = 1e-4,
             datafrac: float = DATAFRAC,
             savemodel=False, savelosses=False,
             savedir='../trained-models/',
             save_interim_models=True, now=NOW,
             loss_func_type=None):
    """
    Using a fraction of training data for quick training and
    trains the model for a few epochs, returning validation loss.
    
    Arguments:
        model: The model to be trained.
        epochs: Number of epochs to train for.
        datafrac: Fraction of training data to use for quick training (default 0.1)
        savemodel: Whether to save the trained model (default False)
        savelosses: Whether to save training and validation losses (default False)
    """
    if train_loader is None or val_loader is None:
        logger.info("Setting up dataloaders since they were not provided.")
        train_loader, val_loader = set_dataloaders(target=model.MODEL_CONFIG.get('target', BASE_MODEL_CONFIG['target']))

    model = model.to(getattr(torch, PRECISION))
    model = model.to(DEVICE)

    if datafrac == 1.0:
        num_train_batches = len(train_loader)
        num_val_batches = len(val_loader)
    else:
        num_train_batches = int( len(train_loader) * datafrac )
        num_train_batches = max(num_train_batches, 1)  # Ensure at least 1 batch is used
        logger.info(f"Using {num_train_batches} batches for training and validation based on data fraction {datafrac} out of {len(train_loader)} input batches.")
        num_val_batches = int( len(val_loader) * datafrac )
        num_val_batches = max(num_val_batches, 1)  # Ensure at least 1 batch is used
        logger.info(f"Using {num_val_batches} batches for validation based on data fraction {datafrac} out of {len(val_loader)} input batches.")

    # Check if model parameters contain NaN or Inf before training
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            logger.warning(f"Parameter {name} contains NaN values before training.")
        if torch.isinf(param).any():
            logger.warning(f"Parameter {name} contains Inf values before training.")
        # logger.info(f"Parameter {name} - min: {param.min().item()}, max: {param.max().item()}, mean: {param.mean().item()}")

    loss_func_type = model.MODEL_CONFIG.get('loss_func_type', None) if loss_func_type is None else loss_func_type

    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=init_lr) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5,   # reduce LR by a factor of 1/2
                patience=4,   # wait for 4 epochs before reducing LR
                threshold=1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # -- Set loss function type and components to track!
    if loss_func_type is None:
        lossfunction = model.loss_function
        lcomps_names = ['recon_loss', 'kl_loss']
    elif loss_func_type == 'mmloss':
        lossfunction = model.mismatch_loss_func
        lcomps_names = ['recon_loss', 'kl_loss', 'mmloss']
    elif loss_func_type == 'mismatch_nokl':
        lossfunction = model.mismatch_nokl_loss_func
        lcomps_names = ['recon_loss', 'mmloss', 'latent_loss']
    elif loss_func_type == 'simple_mse':
        lossfunction = model.simple_mse_loss_func
        lcomps_names = ['recon_loss', 'latent_loss']
    else:
        logger.error(f"Invalid loss function type specified: {loss_func_type}. Using default loss function.")
        lossfunction = model.loss_function
        lcomps_names = ['recon_loss', 'kl_loss']

    # -- Initialize main loss storage
    rloss_train, rloss_eval, rloss_valid = [], [], []
    lcomps_train = {comp_name: [] for comp_name in lcomps_names}
    lcomps_eval = {comp_name: [] for comp_name in lcomps_names}
    lcomps_valid = {comp_name: [] for comp_name in lcomps_names}

    best_val_loss = float('inf')
    for epoch in tqdm(range(epochs)):
        model.train()

        # -- only use tensor updates for GPU efficiency!
        train_loss_log = torch.zeros(num_train_batches, device=DEVICE)
        eval_loss_log = torch.zeros(num_train_batches, device=DEVICE)
        valid_loss_log = torch.zeros(num_val_batches, device=DEVICE)
        lcomps_train_log = {name: torch.zeros(num_train_batches, device=DEVICE) 
                            for name in lcomps_names}
        lcomps_eval_log = {name: torch.zeros(num_train_batches, device=DEVICE)
                            for name in lcomps_names}
        lcomps_valid_log = {name: torch.zeros(num_val_batches, device=DEVICE)
                            for name in lcomps_names}
        
        for idx, databatch in enumerate(tqdm(train_loader, ncols=80, desc="Train-steps")):
            if idx >= num_train_batches:
                break
            
            x, target, labels, keys, strains, attr = databatch
            x, target, labels, keys, strains = x.to(DEVICE), target.to(DEVICE), labels.to(DEVICE), keys.to(DEVICE), strains.to(DEVICE)

            optimizer.zero_grad()
            x_recon, zvars = model(x, labels, keys)

            if loss_func_type is None:
                # -- default loss function only takes 3 arguments!
                loss, *lcomps = lossfunction(target, x_recon, zvars)
            else:
                loss, *lcomps = lossfunction(target, x_recon, zvars, strains, keys, attr)

            loss.backward()
            optimizer.step()

            # -- directly update torch tensor, for GPU efficiency!
            train_loss_log[idx] = loss.detach()

            # -- loss components are returned as tuples of numbers / numpy array!
            # we put the component variable name as the keys in the `lcomps_xx` dict,
            # and store the component values as it is in the corresponding value of the dict.
            # Thus, this works with different loss functions that each return different number of loss components.
            # Later, we can simply iterate over the dictionary and save the losses!
            # NOTE: It is assumed that loss function returns components in order.
            for comp_name, comp_value in zip(lcomps_names, lcomps):
                lcomps_train_log[comp_name][idx] = comp_value.detach()

        train_loss_avg = train_loss_log.mean().item()
        rloss_train.extend(train_loss_log.cpu().numpy().tolist())
        for comp_name in lcomps_names:
            lcomps_train[comp_name].extend(lcomps_train_log[comp_name].cpu().numpy().tolist())
        logger.info(f"Epoch {epoch+1}, Batch Avg Train Loss: {train_loss_avg:.4f}")

        # -- set model to eval mode for validation
        model.eval()

        # -- Do one cycle training in eval mode with no grad after training is finished,
        # -- to compare train and validation losses at the same epoch and check for overfitting etc.
        with torch.no_grad():
            for idx, (x, target, labels, keys, strains, attr) in enumerate(tqdm(train_loader, ncols=80, desc="Train-eval-steps")):
                if idx >= num_train_batches:
                    break
                x, target, labels, keys, strains = x.to(DEVICE), target.to(DEVICE), labels.to(DEVICE), keys.to(DEVICE), strains.to(DEVICE)
                
                x_recon, zvars = model(x, labels, keys)
                
                if loss_func_type is None:
                    # -- default loss function only takes 3 arguments!
                    loss, *lcomps = lossfunction(target, x_recon, zvars)
                else:
                    loss, *lcomps = lossfunction(target, x_recon, zvars, strains, keys, attr)
                
                eval_loss_log[idx] = loss.detach()
                for comp_name, comp_value in zip(lcomps_names, lcomps):
                    lcomps_eval_log[comp_name][idx] = comp_value.detach()

        eval_loss_avg = eval_loss_log.mean().item()
        rloss_eval.extend(eval_loss_log.cpu().numpy().tolist())
        for comp_name in lcomps_names:
            lcomps_eval[comp_name].extend(lcomps_eval_log[comp_name].cpu().numpy().tolist())
        logger.info(f"Epoch {epoch+1}, Batch Avg Train Eval Loss: {eval_loss_avg:.4f}")

        # Evaluate on validation set
        with torch.no_grad():
            for idx, (x, target, labels, keys, strains, attr) in enumerate(tqdm(val_loader, ncols=80, desc="Val-steps")):
                if idx >= num_val_batches:
                    break
                x, target, labels, keys, strains = x.to(DEVICE), target.to(DEVICE), labels.to(DEVICE), keys.to(DEVICE), strains.to(DEVICE)
                
                x_recon, zvars = model(x, labels, keys)

                if loss_func_type is None:
                    # -- default loss function only takes 3 arguments!
                    loss, *lcomps = lossfunction(target, x_recon, zvars)
                else:
                    loss, *lcomps = lossfunction(target, x_recon, zvars, strains, keys, attr)
                
                valid_loss_log[idx] = loss.detach()
                for comp_name, comp_value in zip(lcomps_names, lcomps):
                    lcomps_valid_log[comp_name][idx] = comp_value.detach()

        avg_val_loss = valid_loss_log.mean().item()
        rloss_valid.extend(valid_loss_log.cpu().numpy().tolist())
        for comp_name in lcomps_names:
            lcomps_valid[comp_name].extend(lcomps_valid_log[comp_name].cpu().numpy().tolist())
        logger.info(f"Epoch {epoch+1}, Batch Avg Validation Loss: {avg_val_loss:.4f}")

        # Save model checkpoint at every epoch as backup
        if save_interim_models and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info(f"New best validation loss: {best_val_loss:.4f} at epoch {epoch+1}. Saving model backup.")
            backup_model_path = savedir+f'model-backup-{now}-epoch{epoch}.pt'
            torch.save(model.state_dict(), backup_model_path)
            logger.info(f"Model backup saved at {backup_model_path}")

        # -- TODO: This should be after the validation step?
        scheduler.step(avg_val_loss)

    if savemodel:
        model_path = savedir+f'model-flexcvae-{now}.pt'
        torch.save(model.state_dict(), model_path)
    if savelosses:
        # Save losses to pandas dataframe and then to csv
        train_losses_df = pd.DataFrame({'train_loss': rloss_train})
        eval_losses_df = pd.DataFrame({'train_eval_loss': rloss_eval})
        val_losses_df = pd.DataFrame({'val_loss': rloss_valid})
        for comp_name in lcomps_names:
            train_losses_df[f'train_{comp_name}'] = lcomps_train[comp_name]
            eval_losses_df[f'train_eval_{comp_name}'] = lcomps_eval[comp_name]
            val_losses_df[f'val_{comp_name}'] = lcomps_valid[comp_name]
        train_losses_df.to_csv(savedir+f'losses-flexcvae-train-{now}.csv', index=False)
        eval_losses_df.to_csv(savedir+f'losses-flexcvae-train-eval-{now}.csv', index=False)
        val_losses_df.to_csv(savedir+f'losses-flexcvae-val-{now}.csv', index=False)
    return avg_val_loss


def plot_reconstructions(orig_amp, recon_amp, orig_phase, recon_phase, 
                         orig_hp, recon_hp, orig_hc, recon_hc,
                         title='', savedir='', savename=''):
    """
    Plot the original and reconstructed waveforms for debugging.
    """
    time_arr = calc_time_array(orig_amp.shape[-1])

    plot_twopanel(
        xarr = time_arr.cpu().numpy(),
        yarr = [
            {'Original': orig_amp, 
             'Reconstructed': recon_amp},
            {'Original': orig_phase, 
             'Reconstructed': recon_phase}
        ],
        with_zoom_windows=True,
        title = title,
        axes_labels = ['Time (s)', 'Amplitude', 'Phase (rad)'],
        savename = savedir + savename + f'overplot-ampphase-{NOW}.png',
    )
    logger.info(f"Saved amplitude and phase reconstruction plot at {savedir + savename + f'overplot-ampphase-{NOW}.png'}")

    plot_twopanel(
        xarr = time_arr.cpu().numpy(),
        yarr = [
            {'Original': orig_hp, 
             'Reconstructed': recon_hp},
            {'Original': orig_hc, 
             'Reconstructed': recon_hc}
        ],
        with_zoom_windows=True,
        title = title,
        axes_labels = ['Time (s)', '$h_{+}$', '$h_{\\times}$'],
        savename = savedir + savename + f'overplot-hphc-{NOW}.png',
    )
    logger.info(f"Saved hplus and hcross reconstruction plot at {savedir + savename + f'overplot-hphc-{NOW}.png'}")


def testing(model: {FlexTwoC2E1D, FlexCAE, FlexCAEPhase},
            test_loader=None, savedir=None):
    """
    Test the model on the test dataset and save the results.
    """
    logger.info("Starting testing of the model on the test dataset.")
    if test_loader is None:
        logger.info("Setting up test dataloader since it was not provided.")
        test_loader = set_waveform_dataloaders(return_test_loader=True)
    
    # Initialize dataframe to store mismatch results of whole test set!
    dfmm = pd.DataFrame(columns=[
        'm1', 'm2', 'chi1z', 'chi2z',
        'chirp_mass', 'total_mass', 'mass_ratio',
        'mismatch_amp', 'mismatch_phase', 
        'mismatch_hplus', 'mismatch_hcross',],
        dtype=float)
    
    for idx, databatch in enumerate(tqdm(test_loader, ncols=80, desc="Test-steps")):
        logger.debug(f"Processing test batch {idx+1}/{len(test_loader)}")
        input, target, labels, keys, strains, attr = databatch
        input, target, labels, keys, strains = input.to(DEVICE), target.to(DEVICE), labels.to(DEVICE), keys.to(DEVICE), strains.to(DEVICE)
        
        with torch.no_grad():
            x_recon = model.generate(labels, convert_to_hphc=False)

        mismatch_amp = np.zeros(input.shape[0])
        mismatch_phase = np.zeros(input.shape[0])
        mismatch_hplus = np.zeros(input.shape[0])
        mismatch_hcross = np.zeros(input.shape[0])
        logger.debug(f"Test batch {idx+1}: input shape: {input.shape}, target shape: {target.shape}, labels shape: {labels.shape}, keys shape: {keys.shape}, strains shape: {strains.shape}, x_recon shape: {x_recon.shape}")

        m1s, m2s, chi1zs, chi2zs = labels[:, 0], labels[:, 1], labels[:, 2], labels[:, 3]
        chirpmasses = calc_chirp_mass(m1s, m2s)
        totalmasses = m1s + m2s
        massratios = m1s / m2s
        chieffs = calc_chieff(m1s, m2s, chi1zs, chi2zs)

        for i in range(input.shape[0]):
            recon_amp, recon_phase = x_recon[i, 0, :], x_recon[i, 1, :]
            orig_amp, orig_phase = target[i, 0, :], target[i, 1, :]
            orig_hp, orig_hc = strains[i, 0, :], strains[i, 1, :]
            recon_hp, recon_hc = polarizations_from_amp_phase(recon_amp, recon_phase,
                                                              scale_factor=10**20)

            # -- plot one example of the original and reconstructed waveforms, for debugging!
            if idx == 0 and i == 0:
                plot_reconstructions(
                    orig_amp.cpu().numpy(), recon_amp.cpu().numpy(), 
                    orig_phase.cpu().numpy(), recon_phase.cpu().numpy(), 
                    orig_hp.cpu().numpy(), recon_hp.cpu().numpy(), 
                    orig_hc.cpu().numpy(), recon_hc.cpu().numpy(),
                    title=f'$m_1 = {labels[i, 0].item():.2f}, m_2 = {labels[i, 1].item():.2f}, \\chi_1(z) = {labels[i, 2].item():.2f}, \\chi_2(z) = {labels[i, 3].item():.2f}$',
                    savedir=savedir, savename='test-results-'
                    )

            delta_t = attr['delta_t'][i]
            f_lower = attr['f_lower'][i]

            mismatch_amp[i] = calculate_cosine_distance(recon_amp, orig_amp)
            mismatch_phase[i] = calculate_cosine_distance(recon_phase, orig_phase)
            mismatch_hplus[i] = calc_polarization_mismatch(recon_hp, orig_hp, delta_t, f_lower)
            mismatch_hcross[i] = calc_polarization_mismatch(recon_hc, orig_hc, delta_t, f_lower)

            logger.debug(f"Test batch {idx+1}, sample {i+1}/{input.shape[0]}: m1={m1s[i].item():.2f}, m2={m2s[i].item():.2f}, chi1z={chi1zs[i].item():.2f}, chi2z={chi2zs[i].item():.2f}, chirp_mass={chirpmasses[i].item():.2f}, total_mass={totalmasses[i].item():.2f}, mass_ratio={massratios[i].item():.2f}, chieff={chieffs[i].item():.2f}, mismatch_amp={mismatch_amp[i]:.4e}, mismatch_phase={mismatch_phase[i]:.4e}, mismatch_hplus={mismatch_hplus[i]:.4e}, mismatch_hcross={mismatch_hcross[i]:.4e}")

        dfmm = pd.concat([dfmm, pd.DataFrame({
            'm1': labels[:, 0].cpu().numpy(),
            'm2': labels[:, 1].cpu().numpy(),
            'chi1z': labels[:, 2].cpu().numpy(),
            'chi2z': labels[:, 3].cpu().numpy(),
            'chirp_mass': chirpmasses.flatten().cpu().numpy(),
            'total_mass': totalmasses.flatten().cpu().numpy(),
            'mass_ratio': massratios.flatten().cpu().numpy(),
            'chieff': chieffs.flatten().cpu().numpy(),
            'mismatch_amp': mismatch_amp.flatten(),
            'mismatch_phase': mismatch_phase.flatten(),
            'mismatch_hplus': mismatch_hplus.flatten(),
            'mismatch_hcross': mismatch_hcross.flatten(),
        })], ignore_index=True)
        logger.info(f"Processed test batch {idx+1}/{len(test_loader)}, with average amplitude mismatch {mismatch_amp.mean().item():.4e}, phase mismatch {mismatch_phase.mean().item():.4e}, hplus mismatch {mismatch_hplus.mean().item():.4e}, and hcross mismatch {mismatch_hcross.mean().item():.4e}")

    logger.info(f"Completed testing on {len(test_loader.dataset)} samples.")
    return dfmm



# def objective(trial):
#     """
#     For a 2C2E1D model, suggest hyperparameters, build the model,
#     train for a few epochs, and return validation loss.
#     The number of CNN and FC layers are optimized, along with number
#     of hidden layers in each FC block, base number of CNN channels, etc.
#     Along with these the following parameters are also optimized:
#         - latent_dim
#         - dropout_p
#         - use_batchnorm
#         - activation function
#         - n_cnn / n_fc
#         - n_cnn_enc / n_cnn_dec
#         - n_fc_pre / n_fc_post
#         - base_cnn_channels
#         - pre_fc_sizes / post_fc_sizes
#         - cnn_in_channels / cnn_out_channels
#         - cnn_kernel_size / cnn_dilation / cnn_pool_kernel_size

#     Using a fixed number of trials, the validation loss is minimized.
#     """
#     logger.info("Starting new trial")

#     # Suggest hyperparameters
#     input_dim = 2 * PRESET_ARRAY_SIZE  # Assuming input is a flattened array of shape (2, PRESET_ARRAY_SIZE)
#     num_classes = 4  # Set according to your dataset
#     latent_dim_x = trial.suggest_int("latent_dim_x", 8, 100)
#     latent_dim_key = trial.suggest_int("latent_dim_key", 2, 4)
#     activation = trial.suggest_categorical("activation", ["relu", "silu", "gelu"])

#     dropout_p = trial.suggest_float("dropout_p", 0.1, 0.5)
#     n_cnn_enc = trial.suggest_int("n_cnn_enc", 2, 5)
#     n_cnn_dec = trial.suggest_int("n_cnn_dec", 2, 5)
#     n_fc_pre = trial.suggest_int("n_fc_pre", 1, 3)
#     n_fc_post = trial.suggest_int("n_fc_post", 1, 3)
#     base_cnn_channels = trial.suggest_int("base_cnn_channels", 16, 64)

#     # Suggest only hidden layers, then build full sizes list
#     pre_fc_hidden = trial.suggest_categorical("pre_fc_hidden", [(256,), (512, 256)])
#     # Build pre_fc_sizes and check compatibility
#     pre_fc_sizes_raw = [input_dim] + list(pre_fc_hidden) + [latent_dim_x * 2]
#     # Check for consecutive sizes compatibility
#     pre_fc_sizes = tuple(pre_fc_sizes_raw)
#     n_fc_pre = len(pre_fc_sizes) - 1
#     for i in range(n_fc_pre):
#         if pre_fc_sizes[i] is None or pre_fc_sizes[i+1] is None:
#             import warnings
#             warnings.warn(f"pre_fc_sizes contains None at position {i}: {pre_fc_sizes}")
#         # You could add more checks here for compatibility if needed

#     post_fc_hidden = trial.suggest_categorical("post_fc_hidden", [(128,), (256, 128), (512, 256, 128)])
#     # Set previous layer output to last hidden size plus num_classes for FC block compatibility
#     post_fc_input_size = post_fc_hidden[-1] + num_classes
#     post_fc_sizes_raw = [post_fc_input_size] + list(post_fc_hidden) + [latent_dim_x * 2]
#     post_fc_sizes = tuple(post_fc_sizes_raw)
#     n_fc_post = len(post_fc_sizes) - 1
#     for i in range(n_fc_post):
#         if post_fc_sizes[i] is None or post_fc_sizes[i+1] is None:
#             import warnings
#             warnings.warn(f"post_fc_sizes contains None at position {i}: {post_fc_sizes}")
#         # You could add more checks here for compatibility if needed

#     first_in_channel = trial.suggest_categorical("cnn_first_in_channel", [1, 2])
#     cnn_out_channels = list(trial.suggest_categorical("cnn_out_channels", [(16, 32), (32, 64), (64, 128)]))
#     cnn_in_channels = [first_in_channel] + cnn_out_channels[:-1]
#     cnn_kernel_size = list(trial.suggest_categorical("cnn_kernel_size", [(3, 3), (5, 3), (3, 5)]))
#     cnn_dilation = list(trial.suggest_categorical("cnn_dilation", [(1, 1), (2, 1), (1, 2)]))
#     cnn_pool_kernel_size = list(trial.suggest_categorical("cnn_pool_kernel_size", [(2, 2), (2, 1), (1, 2)]))

#     # Build model with suggested hyperparameters
#     model = TwoC2E1D(
#         labels_mean=params_mean,
#         labels_std=params_std,
#         input_shape=(2, PRESET_ARRAY_SIZE),
#         num_classes=num_classes,
#         encoder_latent_dims=[latent_dim_x, latent_dim_key],
#         cond_dim=[latent_dim_x, latent_dim_key],
#         dropout_p=dropout_p,
#         use_batchnorm=False,
#         activation=activation,
#         n_layers_cnn_encoder=n_cnn_enc,
#         n_layers_cnn_decoder=n_cnn_dec,
#         n_layers_pre_fc=n_fc_pre,
#         n_layers_post_fc=n_fc_post,
#         base_cnn_channels=base_cnn_channels,
#         pre_fc_sizes=pre_fc_sizes,
#         post_fc_sizes=post_fc_sizes,
#         cnn_in_channels=cnn_in_channels,
#         cnn_out_channels=cnn_out_channels,
#         cnn_kernel_size=cnn_kernel_size,
#         cnn_dilation=cnn_dilation,
#         cnn_pool_kernel_size=cnn_pool_kernel_size
#     )

#     logger.info(f"Trial hyperparameters: {trial.params}")
#     val_loss = training(model, epochs=5)
#     return val_loss



def load_flex_model(configpath=None, model_path=None, device=DEVICE, precision=PRECISION):
    """
    Load the trained FlexTwoC2E1D model from the specified path.
    """
    logger.info("Starting training with specified hyperparameters")

    # -- load model to CPU first, if device is not specified!
    if device is None:
        device = torch.device("cpu")
    if precision is None:
        precision = "float32"

    if configpath is not None:
        if not configpath.endswith('.json'):
            configpath += '.json'
        if not os.path.isfile(configpath):
            logger.error(f"Provided MODEL_CONFIG path does not exist: {configpath}")
            raise FileNotFoundError(f"MODEL_CONFIG file not found at {configpath}")
        logger.info(f"Using MODEL_CONFIG: {configpath}")
        MODEL_CONFIG = json.load(open(configpath, 'r'))
    else:
        logger.info("No MODEL_CONFIG provided. Using default hyperparameter values.")
        MODEL_CONFIG = BASE_MODEL_CONFIG.copy()

    # -- Load labels mean and std values if provided in MODEL_CONFIG
    if 'labels_mean' not in MODEL_CONFIG or 'labels_std' not in MODEL_CONFIG:
        logger.warning("labels_mean or labels_std not found in MODEL_CONFIG. Using global params_mean and params_std for normalization.")
        MODEL_CONFIG['labels_mean'] = params_mean
        MODEL_CONFIG['labels_std'] = params_std
    elif isinstance(MODEL_CONFIG['labels_mean'], str) and isinstance(MODEL_CONFIG['labels_std'], str):
        # print(MODEL_CONFIG['labels_std'])
        # print(MODEL_CONFIG['labels_std'].strip('[]').split(','))
        # print(float(MODEL_CONFIG['labels_std'].strip('[]').split(',')[0]))
        # print(type(MODEL_CONFIG['labels_std'].strip('[]').split(',')[0]))
        logger.info("Converting labels_mean and labels_std from str->lists to numpy arrays for model initialization.")
        if MODEL_CONFIG['labels_mean'] == "None" or MODEL_CONFIG['labels_std'] == "None":
            logger.warning("Labels mean or std is None in MODEL_CONFIG, skipping conversion and normalization.")
            MODEL_CONFIG['labels_mean'] = None
            MODEL_CONFIG['labels_std'] = None
        else:
            MODEL_CONFIG['labels_mean'] = np.array(MODEL_CONFIG['labels_mean'].strip('[]').split(',')).astype(float)
            MODEL_CONFIG['labels_std'] = np.array(MODEL_CONFIG['labels_std'].strip('[]').split(',')).astype(float)
    
    if MODEL_CONFIG['labels_mean'] is not None and MODEL_CONFIG['labels_std'] is not None:
        # logger.warning('For now we will use predefined global params_mean and params_std for normalization 
        # instead of converting from MODEL_CONFIG, since the conversion is not working well and giving NaN values 
        # for some reason. This needs to be fixed later.')
        # labels_mean = params_mean.cpu().numpy()
        # labels_std = params_std.cpu().numpy()
        MODEL_CONFIG['labels_mean'] = torch.tensor(MODEL_CONFIG['labels_mean'], dtype=getattr(torch, precision))
        MODEL_CONFIG['labels_std'] = torch.tensor(MODEL_CONFIG['labels_std'], dtype=getattr(torch, precision))

    # Convert some hyperparameters from str to appropriate types if needed (e.g. lists, tuples)
    if isinstance(MODEL_CONFIG['input_shape'], str):
        logger.info("Converting input_shape from str to tuple for model initialization.")
        MODEL_CONFIG['input_shape'] = tuple(map(int, MODEL_CONFIG['input_shape'].strip('()').split(',')))
    if isinstance(MODEL_CONFIG['key_shape'], str):
        logger.info("Converting key_shape from str to tuple for model initialization.")
        MODEL_CONFIG['key_shape'] = tuple(map(int, MODEL_CONFIG['key_shape'].strip('()').split(',')))
    if isinstance(MODEL_CONFIG['target'], str) and MODEL_CONFIG['target'].lower() == 'none':
        logger.info("Setting target to None for model initialization.")
        MODEL_CONFIG['target'] = None

    for k in MODEL_CONFIG:
        if MODEL_CONFIG[k] == "None":
            logger.info(f"Setting {k} to None for model initialization.")
            MODEL_CONFIG[k] = None
        if MODEL_CONFIG[k] == "False":
            logger.info(f"Setting {k} to False for model initialization.")
            MODEL_CONFIG[k] = False
        if MODEL_CONFIG[k] == "True":
            logger.info(f"Setting {k} to True for model initialization.")
            MODEL_CONFIG[k] = True

    if 'modeltype' not in MODEL_CONFIG:
        logger.warning("modeltype not specified in MODEL_CONFIG, defaulting to 'flexcvae'.")
        MODEL_CONFIG['modeltype'] = 'flexcvae'
    if MODEL_CONFIG['modeltype'].lower() not in ['flexcvae', 'flexcae', 'flexcaephase', 'original']:
        logger.error(f"Invalid modeltype specified in MODEL_CONFIG: {MODEL_CONFIG['modeltype']}. Must be 'flexcvae', 'flexcae', 'flexcaephase', or 'original'.")
        raise ValueError(f"Invalid modeltype specified in MODEL_CONFIG: {MODEL_CONFIG['modeltype']}. Must be 'flexcvae', 'flexcae', 'flexcaephase', or 'original'.")

    if MODEL_CONFIG.get('target', None) is not None:
        logger.info(f"Model will be initialized with target: {MODEL_CONFIG['target']}")

    if MODEL_CONFIG['target'] == 'amp_phase':
        logger.info("Initializing FlexCAEPhase model since target is 'amp_phase'.")
        model = FlexCAEPhase(
            MODEL_CONFIG=MODEL_CONFIG,
            input_shape=(2, PRESET_ARRAY_SIZE),
            num_classes=4,
        )
    elif MODEL_CONFIG['modeltype'].lower() == 'flexcae':
        model = FlexCAE(
            MODEL_CONFIG=MODEL_CONFIG,
            input_shape=(2, PRESET_ARRAY_SIZE),
            num_classes=4,
        )
    elif MODEL_CONFIG['modeltype'].lower() == 'original':
        logger.info("Initializing original CVAE model architecture for testing.")
        # -- NOTE: Original trained model `state_dict` doesn't have labels_mean/labels_std, so we do not pass
        # -- the MODEL_CONFIG to CVAE model.
        model = CVAE(input_shape=(2, 8190), 
                     num_classes=4, 
                     key_shape=(2,2))
    else:
        model = FlexTwoC2E1D(
            MODEL_CONFIG=MODEL_CONFIG,
            input_shape=(2, PRESET_ARRAY_SIZE),
            num_classes=4,
        )
    logger.info(f"Model architecture of type {model.__class__.__name__} initialized. Now loading model weights.")
    logger.info(f"Model initialized with the following hyperparameters: {MODEL_CONFIG}")
    # print(model)
    logger.info(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Total number of trainable parameters: \
          {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Send model to device and convert to desired precision
    if device is not None:
        logger.info(f"Moving model to device: {device} and converting to precision: {precision}")
        model.to(getattr(torch, precision))
        model.to(device)
    
    if model_path is None:
        logger.info("No model path provided. Model will be initialized with random weights.")
    else:
        if not os.path.isfile(model_path):
            logger.error(f"Provided model path does not exist: {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
        # -- Check if model weights loaded are of the same precision as our initialized model.
        # -- If not, then convert loaded model to the correct precision before moving to device.
        for name, param in model.named_parameters():
            logger.debug(f"Checking parameter '{name}' of dtype {param.dtype} against desired precision {precision}")
            if param.dtype != getattr(torch, precision):
                logger.info(f"Converting model parameter '{name}' from {param.dtype} to {getattr(torch, precision)} for consistency with initialized model precision.")
                param.data = param.data.to(getattr(torch, precision))
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Loaded model from {model_path}")

    if device.type == 'cuda':
        if precision=='float32':
            torch.set_float32_matmul_precision('high')
        # model = torch.compile(model, mode='max-autotune')  # Compile the model for faster training
        model.compile()  # -- precompile model in default mode for faster training.
        logger.info("Model compiled with torch.compile for faster training.")
    return model


def run_training(configpath=None, model_path=None, fname=None,
                 batch_size=None, num_workers=None,
                 epochs=EPOCHS, datafrac=DATAFRAC):
    """
    Runs training with specified hyperparameters for a single model configuration!
    """
    savedir = f'../trained-models/{NOW}/'
    ensure_dir(savedir)
    if fname is not None:
        savedir += fname+'-'  # add fname to every file name
    model = load_flex_model(configpath=configpath, model_path=model_path)
    logger.debug(model)
    # train_loader, val_loader = set_dataloaders(batch_size=batch_size)
    if batch_size is not None:
        logger.info(f"Using specified batch size: {batch_size}")
    else:
        batch_size = model.MODEL_CONFIG.get('batch_size', BATCH_SIZE)
        logger.info(f"No batch size specified. Using batch size from MODEL_CONFIG: {batch_size}")
    if num_workers is not None:
        logger.info(f"Using specified number of workers: {num_workers}")
    else:
        num_workers = model.MODEL_CONFIG.get('num_workers', 4)
        logger.info(f"No number of workers specified. Using from MODEL_CONFIG: {num_workers}")
    train_loader, val_loader = set_waveform_dataloaders(batch_size=batch_size,
                                                        num_workers=num_workers, 
                                                        target_type='amp_phase',
                                                        input_normalized=True,
                                                        target_normalized=False)
    training(model, 
             epochs=epochs, 
             init_lr=model.MODEL_CONFIG.get('init_lr', 1e-4),
             datafrac=datafrac, 
             train_loader=train_loader, val_loader=val_loader,
             savemodel=True, savelosses=True,
             savedir=savedir)
    logger.info("Training completed and model saved.")
    model.save_model_config(filepath=savedir+f'modelconfig-flexcvae-{NOW}.json',
                             epochs=epochs, datafrac=datafrac)
    logger.info(f"Model configuration saved to {savedir}modelconfig-flexcvae-{NOW}.json")
    train_loader.dataset.close_hdf()  # Close the HDF files after training
    val_loader.dataset.close_hdf()
    print("Training completed and model saved.")



def run_testing(configpath=None, model_path=None, fname=None,
                batch_size=None, num_workers=None,
                results_dir='../v0p1/results/'):
    """
    Runs testing with specified hyperparameters for a trained model!
    """
    savedir = os.path.join(results_dir, f"{TODAY}/", f"test-results-{NOW}/")
    ensure_dir(savedir)

    logger.info(f"Running testing with model config: {configpath}, model weights: {model_path}, and results will be saved to: {savedir}")
    model = load_flex_model(configpath=configpath, model_path=model_path)
    logger.debug(model)
    if batch_size is not None:
        logger.info(f"Using specified batch size: {batch_size}")
    else:
        batch_size = model.MODEL_CONFIG.get('batch_size', BATCH_SIZE)
        logger.info(f"No batch size specified. Using batch size from MODEL_CONFIG: {batch_size}")
    if num_workers is not None:
        logger.info(f"Using specified number of workers: {num_workers}")
    else:
        num_workers = model.MODEL_CONFIG.get('num_workers', 4)
        logger.info(f"No number of workers specified. Using from MODEL_CONFIG: {num_workers}")
    test_loader = set_waveform_dataloaders(batch_size=batch_size,
                                           num_workers=num_workers,
                                           target_type='amp_phase',
                                           input_normalized=True,
                                           target_normalized=False,
                                           return_test_loader=True)
    logger.info(f"Test dataloader set up with {len(test_loader.dataset)} samples and batch size {batch_size}.")

    dfmm = testing(model, test_loader=test_loader, savedir=savedir)

    # -- plot mismatch histograms for amplitude, phase, hplus, and hcross
    from lossplots import plot_mm_hist
    mismatch_types = ['mismatch_amp', 'mismatch_phase', 'mismatch_hplus', 'mismatch_hcross']
    titles = ['Amplitude', 'Phase', '$\\mathbf{h_{+}}$', '$\\mathbf{h_{\\times}}$']
    plot_mm_hist(dfmm, types=mismatch_types, titles=titles, savedir=savedir)
    # save test results and configuration
    dfmm.to_hdf(os.path.join(savedir, f'mismatch-results-{NOW}.h5'), key='dfmm', mode='w')
    config = model.MODEL_CONFIG.copy()
    config.update({
        'training_configpath': configpath,
        'model_path': model_path,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'savedir': savedir,
    })
    with open(os.path.join(savedir, f'test-config-{NOW}.json'), 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f"Test results and configuration saved to {savedir}")
    logger.info("Testing completed and results saved.")


    

def optuna_objective(trial):
    """
    Optuna objective function for hyperparameter optimization of the FlexTwoC2E1D model.
    """
    logger.info("Starting new Optuna trial")
    MODEL_CONFIG = BASE_MODEL_CONFIG.copy()
    # Suggest hyperparameters
    MODEL_CONFIG.update({
        'epochs': 5,  # Keep epochs small for quick Optuna optimization
        'datafrac': 0.3,  # Use 30% of training data for quick training during Optuna optimization
        'batch_size': trial.suggest_categorical("batch_size", [32, 64, 128]),
        'latent_dim_x': trial.suggest_int("latent_dim_x", 8, 128),
        'latent_dim_key': trial.suggest_int("latent_dim_key", 2, 4),
        'activation': trial.suggest_categorical("activation", ["silu", "gelu", "relu"]),
        # some encoder hyperparameters with fixed n_layers for simplicity
        'enc_cnn_in': trial.suggest_categorical("enc_cnn_in", [16, 32, 64]),
        'enc_cnn_out': trial.suggest_categorical("enc_cnn_out", [32, 64, 128]),
        'enc_cnn_kernel': trial.suggest_categorical("enc_cnn_kernel", [3, 4, 5, 6, 7, 8]),
        'enc_cnn_dilation': trial.suggest_categorical("enc_cnn_dilation", [1, 2, 3, 4]),
        'enc_pool_kernel': trial.suggest_categorical("enc_pool_kernel", [2, 3, 4]),
        'enc_postfc_hidden': trial.suggest_categorical("enc_postfc_hidden", [128, 256, 512, 1024]),
        # some decoder hyperparameters with fixed n_layers for simplicity
        'dec_cnn_in': trial.suggest_categorical("dec_cnn_in", [32, 64, 128]),
        'dec_cnn_out': trial.suggest_categorical("dec_cnn_out", [64, 128, 256]),
        'dec_cnn_kernel': trial.suggest_categorical("dec_cnn_kernel", [3, 4, 5, 6, 7, 8]),
        'dec_cnn_dilation': trial.suggest_categorical("dec_cnn_dilation", [1, 2, 3, 4]),
        'dec_pool_kernel': trial.suggest_categorical("dec_pool_kernel", [2, 3, 4]),
        'dec_postfc_hidden': trial.suggest_categorical("dec_postfc_hidden", [128, 256, 512, 1024]),
        'cond_fc_max': trial.suggest_categorical("cond_fc_max", [128, 256, 512, 1024]),
    })

    model = FlexTwoC2E1D(
        MODEL_CONFIG=MODEL_CONFIG,
        input_shape=(2, PRESET_ARRAY_SIZE),
        num_classes=4,
        labels_mean=params_mean,
        labels_std=params_std,
        paramsnorm=True,
    )
    print(model)
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Total number of trainable parameters: \
          {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    # print(model)
    savedir = '../trained-models/optuna/'
    os.makedirs(savedir, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model._save_model_config(filepath=savedir+f'modelconfig-flexcvae-{now}.json',
                             epochs=MODEL_CONFIG['epochs'], 
                             datafrac=MODEL_CONFIG['datafrac'])
    train_loader, val_loader = set_dataloaders(batch_size=MODEL_CONFIG['batch_size'])
    final_val_loss = training(model, epochs=MODEL_CONFIG['epochs'], 
                              datafrac=MODEL_CONFIG['datafrac'], 
                              init_lr=MODEL_CONFIG['init_lr'],
                            train_loader=train_loader, val_loader=val_loader,
                            savemodel=True, savelosses=True, savedir=savedir,
                            save_interim_models=False, now=now)
    logger.info(f"Trial completed with validation loss: {final_val_loss:.4f}")
    # CLEANUP to save GPU memory after each trial
    del model
    # del optimizer
    # Force Python's Garbage Collector
    gc.collect()
    # Clear PyTorch's GPU Cache
    torch.cuda.empty_cache()
    return final_val_loss


def run_optuna():
    study = optuna.create_study(direction="minimize")
    study.optimize(optuna_objective, n_trials=args.trials)
    print("Best trial:", study.best_trial.params)
    # Save the best hyperparameters to a JSON file for future reference
    best_params_path = f"optuna_{args.model_type}_bestparams_{NOW}.json"
    with open(best_params_path, 'w') as f:
        json.dump(study.best_trial.params, f, indent=4)
    logger.info(f"Best hyperparameters saved to {best_params_path}")
    # Save the Optuna study object for future reference
    joblib.dump(study, f"optuna_{args.model_type}_study_{NOW}.pkl")
    logger.info(f"Optuna study saved as optuna_{args.model_type}_study_{NOW}.pkl")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Optuna optimization for TwoC2E1D model")

    parser.add_argument('--model_type', type=str, default='flexcvae', 
                        help="Model type for Optuna or Training study naming")
    parser.add_argument('--trials', type=int, default=20, 
                        help="Number of Optuna trials to run")
    parser.add_argument('--model-config', type=str, default=None,
                        help="Path to JSON file containing model configuration for training")
    parser.add_argument('--model-path', type=str, default=None,
                        help="Path to pre-trained model checkpoint")
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help="Number of epochs for training (default: EPOCHS)")
    parser.add_argument('--label', type=str, default=None,
                        help="Additional label to add to saved model and log filenames for better identification")
    
    parser.add_argument('--batch-size', type=int, default=None,
                        help="Batch size for training (default: from MODEL_CONFIG)")
    parser.add_argument('--num-workers', type=int, default=None,
                        help="Number of workers for data loading (default: from MODEL_CONFIG)")

    methodargs = parser.add_mutually_exclusive_group(required=True)

    methodargs.add_argument('--optuna', action='store_true', 
                        help="Run Optuna optimization")
    methodargs.add_argument('--train', action='store_true', 
                        help="Run training with specified hyperparameters")
    methodargs.add_argument('--test', action='store_true',
                        help="Run testing with specified hyperparameters")
    methodargs.add_argument('--save-data-from-dataset', action='store_true',
                        help="Save input and target data from main datasets for easier loading during training.")
    methodargs.add_argument('--dummyrun', action='store_true',
                        help="Run a dummy training with 10 batches for training loop testing and debugging! 'dummy' also works for this flag, so you can use --dummy or --dummyrun (dunno why?)")
    
    parser = init_verbosity_args(parser)
    args = parser.parse_args()
    init_logging(args, log_dir=f'../{PROJECT_DIR}/logs/{TODAY}')

    print(f'Working on device: {DEVICE}, with precision: {PRECISION}')

    # mp.set_start_method('spawn')

    if args.optuna:
        run_optuna()
    elif args.train:
        logger.info("Running training with specified hyperparameters!")
        run_training(configpath=args.model_config,
                     model_path=args.model_path,
                     epochs=args.epochs, datafrac=1.0, 
                     batch_size=args.batch_size, num_workers=args.num_workers,
                     fname=args.model_type+'-'+args.label if args.label is not None else args.model_type)
    elif args.save_data_from_dataset:
        logger.info("Saving input and target data from main datasets for easier loading during training!")
        save_data_from_dataset()
        
    elif args.test:
        logger.info("Running testing with specified hyperparameters!")
        run_testing(configpath=args.model_config,
                    model_path=args.model_path,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,)
    elif args.dummyrun:
        logger.info("Running dummy training with 10 batches for training loop testing and debugging!")
        run_training(configpath=args.model_config,
                     model_path=args.model_path,
                     batch_size=128,
                     epochs=2, datafrac=0.01, fname='dummyrun-')
