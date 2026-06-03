"""
Model and utilities to calibrate a trained waveform generator model to the target waveform.
We will use the training data to calibrate the generated outputs, by predicting the residuals
between the generated [amp,freq] and the target [amp,freq], for example.
"""

import tqdm
import numpy as np
import logging
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from datacvae import CustomDataset, CustomDataLoader


from optimize import load_flex_model



PROJECT_DIR = 'v0p1'

TODAY = datetime.date.today().strftime("%Y%m%d")
TIME = datetime.datetime.now().strftime("%H%M%S")
NOW = TODAY + '-' + TIME

# -- define some constants for waveform generation
SAMPLE_RATE = 8192  # Hz
DURATION = 1.0  # seconds
FMIN = 20.0  # Hz
FREF = 50.0  # Hz


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    PRECISION = 'float64'  # Use double precision for CUDA if available
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    PRECISION = 'float32'  # Use float32 for MPS since it does not support float64 well
else:
    DEVICE = torch.device("cpu")
    PRECISION = 'float64'  # Use double precision for CPU
logging.info(f"Using device: {DEVICE}, with precision: {PRECISION}")




class ConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=7, dropout=0.0):
        super().__init__()
        padding = kernel_size // 2

        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.GroupNorm(num_groups=8, num_channels=channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.GroupNorm(num_groups=8, num_channels=channels),
        )

        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.net(x))


class ResidualCalibrationCNN(nn.Module):
    """
    A ResNet-style 1D CNN model that takes in some ML generated [amp,freq] and parameters,
    and predicts the residuals to be added to the ML generated outputs to better match the target waveforms.
    """
    def __init__(
        self,
        input_channels,
        output_channels=2,
        hidden_channels=128,
        num_blocks=5,
        kernel_size=7,
        dropout=0.0,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.input_proj = nn.Conv1d(
            input_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.blocks = nn.Sequential(
            *[
                ConvBlock(
                    hidden_channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        self.output_proj = nn.Conv1d(
            hidden_channels,
            output_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

        # Important: initialize final layer near zero
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x):
        """
        x shape: (batch, input_channels, n)
        output shape: (batch, output_channels, n)

        Output is the predicted residual, preferably in standardized/scaled units.
        """
        z = self.input_proj(x)
        z = self.blocks(z)
        residual = self.output_proj(z)
        return residual
    


def get_calibrator_input(wfmodel, originals, labels):
    """
    Function to obtain the input and target for the calibrator model, 
    given the originals waveforms, the parameters, and the trained waveform model.

    Arguments:
    ----------
    wfmodel : torch.nn.Module
        The trained waveform generator model that takes in the parameters and generates [amp,freq].
    originals : torch.Tensor
        The originals waveforms in the form of [amp,freq], shape: (batch, 2, n)
    labels : torch.Tensor
        The parameters corresponding to the waveforms, shape: (batch, num_params)

    Returns:
    --------
    calibrator_input : torch.Tensor
        The input to the calibrator model, which includes the ML generated [amp,freq] and the parameters, 
        shape: (batch, input_channels, n)
    calibrator_target : tuple of torch.Tensor
        The target residuals for amplitude and frequency, each of shape: (batch, 2, n)
    """
    orig_amp, orig_freq = originals[:, 0, :], originals[:, 1, :]

    # -- get the ml predictions for this batch
    with torch.no_grad():
        ml_amp, ml_freq = wfmodel.generate(labels, convert_to_hphc=False)

    target_amp_residual = orig_amp - ml_amp
    target_freq_residual = orig_freq - ml_freq

    assert target_amp_residual.shape == ml_amp.shape == orig_amp.shape
    assert target_freq_residual.shape == ml_freq.shape == orig_freq.shape

    # -- calibrator takes in ml generated [amp,freq] and the parameters, and predicts the residuals
    calibrator_input = torch.stack([ml_amp, ml_freq], dim=1)  # shape: (batch, 2, n)
    param_m1, param_m2, param_s1z, param_s2z = labels[:, 0], labels[:, 1], labels[:, 2], labels[:, 3]

    # -- repeat the parameters across the time dimension to match the shape of ml_amp/ml_freq
    param_m1 = param_m1.unsqueeze(-1).expand(-1, ml_amp.shape[-1])
    param_m2 = param_m2.unsqueeze(-1).expand(-1, ml_amp.shape[-1])
    param_s1z = param_s1z.unsqueeze(-1).expand(-1, ml_amp.shape[-1])
    param_s2z = param_s2z.unsqueeze(-1).expand(-1, ml_amp.shape[-1])

    calibrator_input = torch.cat([calibrator_input, param_m1.unsqueeze(1), param_m2.unsqueeze(1),
                                param_s1z.unsqueeze(1), param_s2z.unsqueeze(1)], dim=1)  # shape: (batch, 6, n)
    return calibrator_input, (target_amp_residual, target_freq_residual)
            
        
    

def train_calibrator(wfmodel_modelpath='../v0p1/trained-models/model-20251004_072338-10', 
                     wfmodel_configpath='modelconfig-cvae-paper-I.json',
                     approximant='SEOBNRv4', batch_size=128, num_epochs=100):

    # -- init waveform model
    wfmodel = load_flex_model(model_path=wfmodel_modelpath, 
                              configpath=wfmodel_configpath, 
                              device=DEVICE, precision=PRECISION,)
    
    trainhdf = '../data/SEOBNRv4-train-100000-fcutoff-uniform-aligned-regen.hdf'
    valhdf = '../data/SEOBNRv4-val-10000-fcutoff-uniform-aligned-regen.hdf'

    calmodel = ResidualCalibrationCNN(
        input_channels=6,  # [ml_amp, ml_freq, param_m1, param_m2, param_s1z, param_s2z]
        output_channels=2,
        hidden_channels=128,
        num_blocks=5,
        kernel_size=7,
        dropout=0.0,
    )

    train_set = CustomDataset(forwhat='train', approximant='', returnattr=True,
                            convert=False, hdf_fname=trainhdf, 
                            train_device=DEVICE, precision=PRECISION)
    logging.info(f'Reading validation data from {valhdf}.hdf')
    valid_set = CustomDataset(forwhat='valid', approximant=approximant, returnattr=True,
                            convert=False, hdf_fname=valhdf, 
                            train_device=DEVICE, precision=PRECISION)
    
    training_loader = CustomDataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = CustomDataLoader(valid_set, batch_size=batch_size, shuffle=True)
    logging.info(training_loader.__dict__)
    ntbatches = len(training_loader)
    nvbatches = len(validation_loader)
    logging.info(f'Number of Training batches: {ntbatches}')  # this doesn't return the batchsize!
    logging.info(f'Number of Validationg batches: {nvbatches}')

    optimizer = torch.optim.AdamW(calmodel.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=20,
    )
    loss_fn = torch.nn.MSELoss()

    for epoch in tqdm(range(num_epochs), desc='Epoch'):
        calmodel.train(True)

        for originals, _, labels, keys, _, attr in tqdm(training_loader, total=len(training_loader), desc='Steps/Batchs'):
            originals = originals.to(DEVICE)
            labels = labels.to(DEVICE)

            calibrator_input, calibrator_target = get_calibrator_input(wfmodel, originals, labels)
            target_amp_residual, target_freq_residual = calibrator_target

            out = calmodel(calibrator_input)  # shape: (batch, 2, n)
            pred_amp_residual, pred_freq_residual = out[:, 0, :], out[:, 1, :]

            # -- compute loss and backprop
            loss = loss_fn(pred_amp_residual, target_amp_residual) + loss_fn(pred_freq_residual, target_freq_residual)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step(loss)

        # validation loop
        calmodel.eval()
        with torch.no_grad():
            val_loss = 0.0
            for originals, _, labels, keys, _, attr in validation_loader:
                originals = originals.to(DEVICE)
                labels = labels.to(DEVICE)

                calibrator_input, calibrator_target = get_calibrator_input(wfmodel, originals, labels)
                target_amp_residual, target_freq_residual = calibrator_target

                out = calmodel(calibrator_input)  # shape: (batch, 2, n)
                pred_amp_residual, pred_freq_residual = out[:, 0, :], out[:, 1, :]

                loss = loss_fn(pred_amp_residual, target_amp_residual) + loss_fn(pred_freq_residual, target_freq_residual)
                val_loss += loss.item()





if __name__ == "__main__":
    main()