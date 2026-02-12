"""
Uses Optuna with `multicvae` for optimization of the
hyper-parameters and number of layers etc.
"""

import os
import datetime
import logging
import joblib

import optuna

import torch
import torch.nn.functional as F

from datacvae import CustomDataset, CustomDataLoader
from multicvae import TwoC2E1D


today = datetime.date.today().strftime("%Y%m%d")
now = datetime.datetime.now().strftime("%H%M%S")
log_filename = f"optuna_multicvae_{today}-{now}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

train_hdf = "data/train_data.h5"
val_hdf = "data/val_data.h5"
train_set = CustomDataset(train_hdf)
val_set = CustomDataset(val_hdf)
train_loader = CustomDataLoader(train_set, batch_size=64, shuffle=True)
val_loader = CustomDataLoader(val_set, batch_size=64, shuffle=False)


def training(model, epochs: int = 5, frac_data: float = 0.1) -> float:
    """
    Using a fraction of training data for quick training and
    trains the model for a few epochs, returning validation loss.
    
    Arguments:
        model: The model to be trained.
        epochs: Number of epochs to train for.
        frac_data: Fraction of training data to use for quick training (default 0.1).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_train_batches = int(len(train_loader) * frac_data)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (x, cond) in enumerate(train_loader):
            if batch_idx >= num_train_batches:
                break
            x, cond = x.to(device), cond.to(device)
            optimizer.zero_grad()
            x_recon, zvars = model(x, cond)
            loss, recon_loss, kl_loss = model.loss_function(x_recon, x, zvars)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / num_train_batches
        logging.info(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, cond in val_loader:
            x, cond = x.to(device), cond.to(device)
            x_recon, zvars = model(x, cond)
            loss, recon_loss, kl_loss = model.loss_function(x_recon, x, zvars)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    logging.info(f"Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss


def objective(trial):
    """
    For a 2C2E1D model, suggest hyperparameters, build the model,
    train for a few epochs, and return validation loss.
    The number of CNN and FC layers are optimized, along with number
    of hidden layers in each FC block, base number of CNN channels, etc.
    Along with these the following parameters are also optimized:
        - latent_dim
        - dropout_p
        - use_batchnorm
        - activation function
        - n_cnn / n_fc
        - n_cnn_enc / n_cnn_dec
        - n_fc_pre / n_fc_post
        - base_cnn_channels
        - pre_fc_sizes / post_fc_sizes
        - cnn_in_channels / cnn_out_channels
        - cnn_kernel_size / cnn_dilation / cnn_pool_kernel_size

    Using a fixed number of trials, the validation loss is minimized.
    """
    logging.info("Starting new trial")
    # Suggest hyperparameters
    latent_dim = trial.suggest_int("latent_dim", 10, 100)
    dropout_p = trial.suggest_float("dropout_p", 0.1, 0.5)
    use_batchnorm = trial.suggest_categorical("use_batchnorm", [True, False])
    activation = trial.suggest_categorical("activation", ["relu", "leaky_relu", "elu"])
    n_cnn_enc = trial.suggest_int("n_cnn_enc", 2, 5)
    n_cnn_dec = trial.suggest_int("n_cnn_dec", 2, 5)
    n_fc_pre = trial.suggest_int("n_fc_pre", 1, 3)
    n_fc_post = trial.suggest_int("n_fc_post", 1, 3)
    base_cnn_channels = trial.suggest_int("base_cnn_channels", 16, 64)

    pre_fc_sizes = trial.suggest_categorical("pre_fc_sizes", [[128], [256, 128], [512, 256, 128]])
    post_fc_sizes = trial.suggest_categorical("post_fc_sizes", [[128], [256, 128], [512, 256, 128]])
    cnn_in_channels = trial.suggest_categorical("cnn_in_channels", [[1, 16], [1, 32], [1, 64]])
    cnn_out_channels = trial.suggest_categorical("cnn_out_channels", [[16, 32], [32, 64], [64, 128]])
    cnn_kernel_size = trial.suggest_categorical("cnn_kernel_size", [[3, 3], [5, 3], [3, 5]])
    cnn_dilation = trial.suggest_categorical("cnn_dilation", [[1, 1], [2, 1], [1, 2]])
    cnn_pool_kernel_size = trial.suggest_categorical("cnn_pool_kernel_size", [[2, 2], [2, 1], [1, 2]])

    # Build model with suggested hyperparameters
    model = TwoC2E1D(
        input_dim=8191,  
        condition_dim=4,  
        latent_dim=latent_dim,
        dropout_p=dropout_p,
        use_batchnorm=use_batchnorm,
        activation=activation,
        n_cnn_enc=n_cnn_enc,
        n_cnn_dec=n_cnn_dec,
        n_fc_pre=n_fc_pre,
        n_fc_post=n_fc_post,
        base_cnn_channels=base_cnn_channels,
        pre_fc_sizes=pre_fc_sizes,
        post_fc_sizes=post_fc_sizes,
        cnn_in_channels=cnn_in_channels,
        cnn_out_channels=cnn_out_channels,
        cnn_kernel_size=cnn_kernel_size,
        cnn_dilation=cnn_dilation,
        cnn_pool_kernel_size=cnn_pool_kernel_size
    )

    logging.info(f"Trial hyperparameters: {trial.params}")
    val_loss = training(model, epochs=5)
    return val_loss


if __name__ == "__main__":

    # Run Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    print("Best trial:", study.best_trial.params)
    # Save the study for future reference
    joblib.dump(study, f"optuna_multicvae_study_{today}-{now}.pkl")
