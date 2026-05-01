"""
Uses Optuna with `multicvae` for optimization of the
hyper-parameters and number of layers etc.
"""

import os
import gc
import json
import argparse
import pandas as pd
import datetime
import logging
import joblib

from tqdm import tqdm

import optuna

import torch
import torch.nn.functional as F
import numpy as np

from datacvae import CustomDataset, CustomDataLoader
# from multicvae import TwoC2E1D

from flexcvae import FlexTwoC2E1D, FlexCAE, FlexCAEPhase


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
    PRECISION = 'float64'  # Use double precision for CUDA if available
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    PRECISION = 'float32'  # Use float32 for MPS since it does not support float64 well
else:
    DEVICE = torch.device("cpu")
    PRECISION = 'float64'  # Use double precision for CPU
print(f"Using device: {DEVICE}, with precision: {PRECISION}")

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

# -- get mean and std of labels for normalization
params_fname = '../data/params-' + APPROXIMANT + '-train-100000-fcutoff-uniform-aligned-regen'
params_df = pd.read_csv(params_fname+'.csv', index_col=0, sep=',')
params_mean = params_df.mean().values
params_std = params_df.std().values
logging.info(f"Labels mean: {params_mean}")
logging.info(f"Labels std: {params_std}")
params_mean = torch.tensor(params_mean, dtype=getattr(torch, PRECISION)).to(DEVICE)
params_std = torch.tensor(params_std, dtype=getattr(torch, PRECISION)).to(DEVICE)


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
    logging.info(f'Reading training data from {train_hdf}.hdf')
    train_set = CustomDataset(forwhat='train', approximant=APPROXIMANT, returnattr=True,
                            hdf_fname=train_hdf, train_device=DEVICE, precision=PRECISION,
                            target=target)
    logging.info(f'Reading validation data from {val_hdf}.hdf')
    valid_set = CustomDataset(forwhat='valid', approximant=APPROXIMANT, returnattr=True,
                            hdf_fname=val_hdf, train_device=DEVICE, precision=PRECISION, 
                            target=target)
    train_loader = CustomDataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = CustomDataLoader(valid_set, batch_size=batch_size, shuffle=False)
    logging.info(f"Training dataset size: {len(train_set)}, Validation dataset size: {len(valid_set)}")
    return train_loader, val_loader

def training(model: {FlexTwoC2E1D, FlexCAE, FlexCAEPhase}, 
             train_loader=None, val_loader=None,
             epochs: int = 5, 
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
    os.makedirs(savedir, exist_ok=True)
    if train_loader is None or val_loader is None:
        logging.info("Setting up dataloaders since they were not provided.")
        train_loader, val_loader = set_dataloaders(target=model.MODEL_CONFIG.get('target', BASE_MODEL_CONFIG['target']))

    model = model.to(getattr(torch, PRECISION))
    model = model.to(DEVICE)

    if datafrac == 1.0:
        num_train_batches = len(train_loader)
        num_val_batches = len(val_loader)
    else:
        num_train_batches = int(len(train_loader) * datafrac) // train_loader.batch_size
        num_val_batches = int(len(val_loader) * datafrac) // val_loader.batch_size
    logging.info(f"Using {num_train_batches} batches for training and validation based on data fraction {datafrac} out of {len(train_loader)} data inputs.")

    # Check if model parameters contain NaN or Inf before training
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            logging.warning(f"Parameter {name} contains NaN values before training.")
        if torch.isinf(param).any():
            logging.warning(f"Parameter {name} contains Inf values before training.")
        # logging.info(f"Parameter {name} - min: {param.min().item()}, max: {param.max().item()}, mean: {param.mean().item()}")

    loss_func_type = model.MODEL_CONFIG.get('loss_func_type', None) if loss_func_type is None else loss_func_type

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) 
    # -- NOTE: ReduceLROnPlateau is not ideal for our use, since it
    # -- reduces the darn LR too quickly and then we don't get much training done.
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #             optimizer, 
    #             mode='min', 
    #             factor=0.5, 
    #             patience=2, 
    #             threshold=1e-7)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # -- Set loss function type and components to track!
    rloss_train, rloss_eval, rloss_val = [], [], []
    lcomps_train, lcomps_eval, lcomps_val = {}, {}, {}
    if loss_func_type is None:
        lossfunction = model.loss_function
        lcomps_names = ['recon_loss', 'kl_loss']
    elif loss_func_type == 'mismatch_nokl':
        lossfunction = model.mismatch_nokl_loss_func
        lcomps_names = ['recon_loss', 'mmloss']
    elif loss_func_type == 'mmloss':
        lossfunction = model.mismatch_loss_func
        lcomps_names = ['recon_loss', 'kl_loss', 'mmloss']
    else:
        logging.error(f"Invalid loss function type specified: {loss_func_type}. Using default loss function.")
        lossfunction = model.loss_function
        lcomps_names = ['recon_loss', 'kl_loss']
    # -- Initialize loss component dictionaries!
    for comp_name in lcomps_names:
        lcomps_train[comp_name] = []
        lcomps_eval[comp_name] = []
        lcomps_val[comp_name] = []

    # Train for a few epochs
    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        for idx, (x, target, labels, keys, strains, attr) in enumerate(tqdm(train_loader, ncols=80, desc="Train-steps")):
            if idx >= num_train_batches:
                break
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
            train_loss += loss.item()
            rloss_train.append(loss.item())
            # -- loss components are returned as tuples of numbers / numpy array!
            # we put the component variable name as the keys in the `lcomps_xx` dict,
            # and store the component values as it is in the corresponding value of the dict.
            # Thus, this works with different loss functions that each return different number of loss components.
            # Later, we can simply iterate over the dictionary and save the losses!
            # NOTE: It is assumed that loss function returns components in the following order:
            # total_loss, recon_loss, kl_loss, mismatch_loss (if applicable), but the component names are taken from 
            # `lcomps_names` list which is set according to the loss function type.
            for comp_name, comp_value in zip(lcomps_names, lcomps):
                lcomps_train[comp_name].append(comp_value.item())

        avg_train_loss = train_loss / num_train_batches
        logging.info(f"Epoch {epoch+1}, Batch Avg Train Loss: {avg_train_loss:.4f}")

        # -- TODO: This should be after the validation step?
        scheduler.step()

        # Save model checkpoint at every epoch as backup
        if save_interim_models:
            backup_model_path = savedir+f'model-backup-{now}-epoch{epoch}.pt'
            torch.save(model.state_dict(), backup_model_path)
            logging.info(f"Model backup saved at {backup_model_path}")

        # -- set model to eval mode for validation
        model.eval()

        # -- Do one cycle training in eval mode with no grad after training is finished,
        # -- to compare train and validation losses at the same epoch and check for overfitting etc.
        train_eval_loss = 0.0
        with torch.no_grad():
            for idx, (x, target, labels, keys, strains) in enumerate(tqdm(train_loader, ncols=80, desc="Train-eval-steps")):
                if idx >= num_train_batches:
                    break
                x, target, labels, keys, strains = x.to(DEVICE), target.to(DEVICE), labels.to(DEVICE), keys.to(DEVICE), strains.to(DEVICE)
                x_recon, zvars = model(x, labels, keys)
                if loss_func_type is None:
                    # -- default loss function only takes 3 arguments!
                    loss, *lcomps = lossfunction(target, x_recon, zvars)
                else:
                    loss, *lcomps = lossfunction(target, x_recon, zvars, strains, keys, attr)
                train_eval_loss += loss.item()
                for comp_name, comp_value in zip(lcomps_names, lcomps):
                    lcomps_eval[comp_name].append(comp_value.item())
        avg_train_eval_loss = train_eval_loss / num_train_batches
        logging.info(f"Epoch {epoch+1}, Batch Avg Train Eval Loss: {avg_train_eval_loss:.4f}")

        # Evaluate on validation set
        val_loss = 0.0
        with torch.no_grad():
            for idx, (x, target, labels, keys, strains) in enumerate(tqdm(val_loader, ncols=80, desc="Val-steps")):
                if idx >= num_val_batches:
                    break
                x, target, labels, keys, strains = x.to(DEVICE), target.to(DEVICE), labels.to(DEVICE), keys.to(DEVICE), strains.to(DEVICE)
                x_recon, zvars = model(x, labels, keys)
                if loss_func_type is None:
                    # -- default loss function only takes 3 arguments!
                    loss, *lcomps = lossfunction(target, x_recon, zvars)
                else:
                    loss, *lcomps = lossfunction(target, x_recon, zvars, strains, keys, attr)
                val_loss += loss.item()
                for comp_name, comp_value in zip(lcomps_names, lcomps):
                    lcomps_val[comp_name].append(comp_value.item())
        avg_val_loss = val_loss / num_val_batches
        logging.info(f"Epoch {epoch+1}, Batch Avg Validation Loss: {avg_val_loss:.4f}")

    if savemodel:
        model_path = savedir+f'model-flexcvae-{now}.pt'
        torch.save(model.state_dict(), model_path)
    if savelosses:
        # Save losses to pandas dataframe and then to csv
        train_losses_df = pd.DataFrame({'train_loss': rloss_train})
        eval_losses_df = pd.DataFrame({'train_eval_loss': lcomps_eval[lcomps_names[0]]})
        val_losses_df = pd.DataFrame({'val_loss': rloss_val})
        for comp_name in lcomps_names:
            train_losses_df[f'train_{comp_name}'] = lcomps_train[comp_name]
            eval_losses_df[f'train_eval_{comp_name}'] = lcomps_eval[comp_name]
            val_losses_df[f'val_{comp_name}'] = lcomps_val[comp_name]
        train_losses_df.to_csv(savedir+f'losses-flexcvae-train-{now}.csv', index=False)
        eval_losses_df.to_csv(savedir+f'losses-flexcvae-train-eval-{now}.csv', index=False)
        val_losses_df.to_csv(savedir+f'losses-flexcvae-val-{now}.csv', index=False)
    return avg_val_loss


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
#     logging.info("Starting new trial")

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

#     logging.info(f"Trial hyperparameters: {trial.params}")
#     val_loss = training(model, epochs=5)
#     return val_loss



def load_flex_model(configpath=None, model_path=None):
    """
    Load the trained FlexTwoC2E1D model from the specified path.
    """
    logging.info("Starting training with specified hyperparameters")
    if configpath is not None:
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
    if 'labels_mean' not in MODEL_CONFIG or 'labels_std' not in MODEL_CONFIG:
        logging.warning("labels_mean or labels_std not found in MODEL_CONFIG. Using global params_mean and params_std for normalization.")
        MODEL_CONFIG['labels_mean'] = params_mean
        MODEL_CONFIG['labels_std'] = params_std
    elif isinstance(MODEL_CONFIG['labels_mean'], str) and isinstance(MODEL_CONFIG['labels_std'], str):
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
    if MODEL_CONFIG['modeltype'].lower() not in ['flexcvae', 'flexcae']:
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
    elif MODEL_CONFIG.get('modeltype', 'flexcvae').lower() == 'flexcae':
        model = FlexCAE(
            MODEL_CONFIG=MODEL_CONFIG,
            input_shape=(2, PRESET_ARRAY_SIZE),
            num_classes=4,
        )
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
    
    if model_path is None:
        logging.info("No model path provided. Model will be initialized with random weights.")
    else:
        if not os.path.isfile(model_path):
            logging.error(f"Provided model path does not exist: {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
        # -- Check if model weights loaded are of the same precision as our initialized model.
        # -- If not, then convert loaded model to the correct precision before moving to device.
        for name, param in model.named_parameters():
            if param.dtype != getattr(torch, PRECISION):
                logging.info(f"Converting model parameter '{name}' from {param.dtype} to {getattr(torch, PRECISION)} for consistency with initialized model precision.")
                param.data = param.data.to(getattr(torch, PRECISION))
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        logging.info(f"Loaded model from {model_path}")
    # NOTE: Model is moved to desired device and precision during training!
    return model


def run_training(configpath=None, model_path=None,
                 batch_size=BATCH_SIZE, epochs=EPOCHS, datafrac=DATAFRAC):
    """
    Runs training with specified hyperparameters for a single model configuration!
    """
    model = load_flex_model(configpath=configpath, model_path=model_path)
    logging.debug(model)
    train_loader, val_loader = set_dataloaders(batch_size=batch_size)
    training(model, epochs=epochs, datafrac=datafrac, 
             train_loader=train_loader, val_loader=val_loader,
             savemodel=True, savelosses=True)
    model._save_model_config(filepath=f'../trained-models/modelconfig-flexcvae-{NOW}.json',
                             epochs=epochs, datafrac=datafrac)
    

def optuna_objective(trial):
    """
    Optuna objective function for hyperparameter optimization of the FlexTwoC2E1D model.
    """
    logging.info("Starting new Optuna trial")
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
                            train_loader=train_loader, val_loader=val_loader,
                            savemodel=True, savelosses=True, savedir=savedir,
                            save_interim_models=False, now=now)
    logging.info(f"Trial completed with validation loss: {final_val_loss:.4f}")
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
    logging.info(f"Best hyperparameters saved to {best_params_path}")
    # Save the Optuna study object for future reference
    joblib.dump(study, f"optuna_{args.model_type}_study_{NOW}.pkl")
    logging.info(f"Optuna study saved as optuna_{args.model_type}_study_{NOW}.pkl")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Optuna optimization for TwoC2E1D model")
    parser.add_argument('--optuna', action='store_true', 
                        help="Run Optuna optimization")
    parser.add_argument('--model_type', type=str, default='flexcvae', 
                        help="Model type for Optuna or Training study naming")
    parser.add_argument('--trials', type=int, default=20, 
                        help="Number of Optuna trials to run")
    parser.add_argument('--train', action='store_true', 
                        help="Run training with specified hyperparameters")
    parser.add_argument('--model-config', type=str, default=None,
                        help="Path to JSON file containing model configuration for training")
    parser.add_argument('--model-path', type=str, default=None,
                        help="Path to pre-trained model checkpoint")

    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Enable verbose logging")
    parser.add_argument('-d', '--debug', action='store_true',
                        help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    logfname = f"optimize-{args.model_type}-{NOW}.log"
    log_dir = f'../logs/{TODAY}/'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, logfname)
    logging.basicConfig(
        format='%(asctime)s: %(levelname)s: %(message)s',
        level=log_level,
        datefmt='%y-%m-%d %H:%M:%S',
        force=True,
        handlers=[
            logging.StreamHandler(),  # Log to console
            logging.FileHandler(log_file)  # Log to file
        ]
    )
    
    # Set FileHandler to always be at least INFO level
    for handler in logging.root.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setLevel(max(handler.level, logging.INFO))

    if args.optuna:
        run_optuna()
    if args.train:
        run_training(configpath=args.model_config,
                     model_path=args.model_path,
                     epochs=10, datafrac=1.0)
