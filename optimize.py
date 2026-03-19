"""
Uses Optuna with `multicvae` for optimization of the
hyper-parameters and number of layers etc.
"""

import os
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

from datacvae import CustomDataset, CustomDataLoader
# from multicvae import TwoC2E1D
from flexcvae import TwoC2E1D as FlexTwoC2E1D


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
    'activation_name': 'gelu',
    'target': 'amp_phase',  # default target is normed amp-freq, but can be set to 'logamp_phase' for log-amp and phase target
}

datadir = "../data/"
train_hdf = datadir + 'SEOBNRv4-train-100000-fcutoff-uniform-aligned-regen'
val_hdf = datadir + "SEOBNRv4-val-100000-fcutoff-uniform-aligned-regen"

logging.info(f'Reading training data from {train_hdf}.hdf')
train_set = CustomDataset(forwhat='train', approximant=APPROXIMANT, returnattr=False,
                        hdf_fname=train_hdf, train_device=DEVICE, precision=PRECISION,
                        target=BASE_MODEL_CONFIG['target'])
logging.info(f'Reading validation data from {val_hdf}.hdf')
valid_set = CustomDataset(forwhat='valid', approximant=APPROXIMANT, returnattr=False,
                        hdf_fname=val_hdf, train_device=DEVICE, precision=PRECISION, 
                        target=BASE_MODEL_CONFIG['target'])

# -- get mean and std of labels for normalization
params_fname = '../data/params-' + APPROXIMANT + '-train-100000-fcutoff-uniform-aligned-regen'
params_df = pd.read_csv(params_fname+'.csv', index_col=0, sep=',')
params_mean = params_df.mean().values
params_std = params_df.std().values
logging.info(f"Labels mean: {params_mean}")
logging.info(f"Labels std: {params_std}")
params_mean = torch.tensor(params_mean, dtype=getattr(torch, PRECISION)).to(DEVICE)
params_std = torch.tensor(params_std, dtype=getattr(torch, PRECISION)).to(DEVICE)


def set_dataloaders(batch_size=BATCH_SIZE):
    train_loader = CustomDataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = CustomDataLoader(valid_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def training(model: FlexTwoC2E1D, 
             train_loader=None, val_loader=None,
             epochs: int = 5, 
             datafrac: float = 0.1,
             savemodel=False, savelosses=False,
             savedir='../trained_models/'):
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
        train_loader, val_loader = set_dataloaders()
    logging.info(f"Starting training for {epochs} epochs with data fraction {datafrac}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = model.to(getattr(torch, PRECISION))

    # Check if model parameters contain NaN or Inf before training
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            logging.warning(f"Parameter {name} contains NaN values before training.")
        if torch.isinf(param).any():
            logging.warning(f"Parameter {name} contains Inf values before training.")
        # logging.info(f"Parameter {name} - min: {param.min().item()}, max: {param.max().item()}, mean: {param.mean().item()}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5, 
                patience=2, 
                threshold=1e-7)
    num_train_batches = int(len(train_loader) * datafrac)

    # Train for a few epochs
    rloss_train, rloss_recon, rloss_kl = [], [], []
    rloss_val, rloss_recon_val, rloss_kl_val = [], [], []
    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        for batch_idx, (x, target, labels, keys, strains) in enumerate(tqdm(train_loader, ncols=80, desc="Train-steps")):
            if batch_idx >= num_train_batches:
                break
            x, target, labels, keys = x.to(device), target.to(device), labels.to(device), keys.to(device)
            optimizer.zero_grad()
            x_recon, zvars = model(x, labels, keys)
            loss, recon_loss, kl_loss = model.loss_function(target, x_recon, zvars)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            rloss_train.append(loss.item())
            rloss_recon.append(recon_loss.item())
            rloss_kl.append(kl_loss.item())
        avg_train_loss = train_loss / num_train_batches
        logging.info(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")
        scheduler.step(avg_train_loss)

        # Save model checkpoint at every epoch as backup
        backup_model_path = savedir+f'model-backup-{NOW}epoch{epoch}.pt'
        torch.save(model.state_dict(), backup_model_path)
        logging.info(f"Model backup saved at {backup_model_path}")

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, target, labels, keys, strains in tqdm(val_loader, ncols=80, desc="Val-steps"):
                x, target, labels, keys = x.to(device), target.to(device), labels.to(device), keys.to(device)
                x_recon, zvars = model(x, labels, keys)
                loss, recon_loss, kl_loss = model.loss_function(target, x_recon, zvars)
                val_loss += loss.item()
                rloss_val.append(loss.item())
                rloss_recon_val.append(recon_loss.item())
                rloss_kl_val.append(kl_loss.item())
        avg_val_loss = val_loss / len(val_loader)
        logging.info(f"Validation Loss: {avg_val_loss:.4f}")

    if savemodel:
        model_path = savedir+f'model-flexcvae-{NOW}.pt'
        torch.save(model.state_dict(), model_path)
    if savelosses:
        # Save losses to pandas dataframe and then to csv
        train_losses_df = pd.DataFrame({
            'train_loss': rloss_train,
            'recon_loss': rloss_recon,
            'kl_loss': rloss_kl,
        })
        val_losses_df = pd.DataFrame({
            'val_loss': rloss_val,
            'val_recon_loss': rloss_recon_val,
            'val_kl_loss': rloss_kl_val,
        })
        train_losses_df.to_csv(savedir+f'losses-flexcvae-train-{NOW}.csv', index=False)
        val_losses_df.to_csv(savedir+f'losses-flexcvae-val-{NOW}.csv', index=False)
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

def run_optuna():
    study = optuna.create_study(direction="minimize")
    study.optimize(run_training, n_trials=args.trials)
    print("Best trial:", study.best_trial.params)
    joblib.dump(study, f"optuna_{args.model_type}_study_{NOW}.pkl")


def run_training(configpath=None, batch_size=BATCH_SIZE, epochs=EPOCHS, datafrac=DATAFRAC):
    logging.info("Starting training with specified hyperparameters")
    if configpath is not None:
        logging.info(f"Using MODEL_CONFIG: {configpath}")
        MODEL_CONFIG = json.load(open(configpath, 'r'))
    else:
        logging.info("No MODEL_CONFIG provided. Using default hyperparameter values.")
        MODEL_CONFIG = BASE_MODEL_CONFIG
    model = FlexTwoC2E1D(
        MODEL_CONFIG=MODEL_CONFIG,
        input_shape=(2, PRESET_ARRAY_SIZE),
        num_classes=4,
        labels_mean=params_mean,
        labels_std=params_std,
        paramsnorm=True,
    )
    # print(model)
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
          )
    # print(model)
    model._save_model_config(filepath=f'../trained-models/modelconfig-flexcvae-{NOW}.json',
                             epochs=epochs, datafrac=datafrac)
    train_loader, val_loader = set_dataloaders(batch_size=batch_size)
    training(model, epochs=epochs, datafrac=datafrac, 
             train_loader=train_loader, val_loader=val_loader,
             savemodel=True, savelosses=True)
    

def optuna_objective(trial):
    MODEL_CONFIG = BASE_MODEL_CONFIG.copy()
    # Suggest hyperparameters
    MODEL_CONFIG.update({
        'epochs': 5,
        'datafrac': 0.5,
        'batch_size': trial.suggest_categorical("batch_size", [32, 64, 128]),
        'latent_dim_x': trial.suggest_int("latent_dim_x", 8, 128),
        'latent_dim_key': trial.suggest_int("latent_dim_key", 2, 4),
        'activation_name': trial.suggest_categorical("activation_name", ["silu", "gelu"]),
    })

    model = FlexTwoC2E1D(
        MODEL_CONFIG=MODEL_CONFIG,
        input_shape=(2, PRESET_ARRAY_SIZE),
        num_classes=4,
        labels_mean=params_mean,
        labels_std=params_std,
        paramsnorm=True,
    )
    # print(model)
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
          )
    # print(model)
    savedir = '../trained-models/optuna/'
    os.makedirs(savedir, exist_ok=True)
    model._save_model_config(filepath=savedir+f'modelconfig-flexcvae-{NOW}.json',
                             epochs=MODEL_CONFIG['epochs'], datafrac=MODEL_CONFIG['datafrac'])
    train_loader, val_loader = set_dataloaders(batch_size=MODEL_CONFIG['batch_size'])
    final_val_loss = training(model, epochs=MODEL_CONFIG['epochs'], datafrac=MODEL_CONFIG['datafrac'], 
                train_loader=train_loader, val_loader=val_loader,
                savemodel=True, savelosses=True, savedir=savedir)
    return final_val_loss


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
    parser.add_argument('--model_config', type=str, default=None,
                        help="Path to JSON file containing model configuration for training")

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
        format='%(levelsize)s | %(asctime)s: %(message)s',
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
                     epochs=15, datafrac=1.0)