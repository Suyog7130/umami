import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal, kl_divergence

import matplotlib.pyplot as plt

import numpy as np
from datetime import datetime


from utils.gwutils import polarizations_from_ampfreq, calc_polarization_mismatch


class XEncoder(nn.Module):
    def __init__(self, latent_dim_x=8, num_classes=2, input_shape=(28, 28, 1)):
        super(XEncoder, self).__init__()
        self.input_shape = input_shape
        self.latent_dim_x = latent_dim_x
        self.num_classes = num_classes

        # Fully connected layers (NN layers)
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(torch.prod(torch.tensor(self.input_shape))), 500),  # First NN layer
            nn.ReLU()
        )

        # CNN layers
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=5, kernel_size=16, dilation=1),
            nn.MaxPool1d(kernel_size=4),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=15, kernel_size=16, dilation=1),
            nn.MaxPool1d(kernel_size=4),
            nn.ReLU(),
        )

        # Fully connected layers (NN layers after CNN)
        self.fc2 = nn.Sequential(
            nn.Linear(15 * self._calculate_cnn_output_size(self.input_shape[0]*self.input_shape[1]) + self.num_classes, 500),  # Second NN layer
            nn.ReLU(),
            nn.Linear(500, 500),  # Third NN layer
            nn.ReLU(),
            nn.Linear(500, latent_dim_x * 2),  # Output layer
            nn.ReLU()
        )
    
    def _calculate_cnn_output_size(self, sequence_length=500):
        """
        Calculates the output size of the CNN layer
        s after applying convolutions and pooling.

        Args:
            sequence_length (int): The input sequence length to the CNN layers.

        Returns:
            int: The size of the flattened CNN output.
        """
        logging.debug(f'sequence_length={sequence_length}')
        # First CNN layer
        sequence_length = ((sequence_length - (16 - 1) - 1) // 1 + 1)  # Conv1
        sequence_length = (sequence_length - 4) // 4 + 1  # Pool1

        # Second CNN layer
        sequence_length = ((sequence_length - (16 - 1) - 1) // 1 + 1)  # Conv2
        sequence_length = (sequence_length - 4) // 4 + 1  # Pool2
        logging.debug(f'sequence_length={sequence_length}')
        return sequence_length

    def forward(self, x, labels):
        """
        Forward pass through the x_encoder.

        Args:
            x (Tensor): Input tensor with shape [batch_size, *input_shape].
            labels (Tensor): Conditional labels with shape [batch_size, num_classes].

        Returns:
            Latent space representation for the strains input.
        """
        # print(x.shape)

        # Pass through the first fully connected layer
        # Remove the NN layer before CNN layers.
        # x = self.fc1(x)

        # Reshape for CNN layers
        x = x.view(x.size(0), 1, -1)  # Reshape to [batch_size, channels, sequence_length]

        # Pass through CNN layers
        logging.debug(x.shape)
        x = self.conv1(x)
        x = self.conv2(x)

        # Flatten the output of CNN layers
        logging.debug(x.shape)
        x = x.view(x.size(0), -1) # Flatten to [batch_size, num_features]

        # Concatenate input and labels
        # print(x.shape)
        x = torch.cat([x.view(x.size(0), -1), labels], dim=1)
        logging.debug(x.shape)

        # Flatten and pass through the second fully connected layer
        x = self.fc2(x)
        logging.debug(x.shape)
        logging.debug('---')

        return x.view(-1, self.latent_dim_x * 2)


class KeyEncoder(nn.Module):
    def __init__(self, latent_dim_key=3, num_classes=2, input_shape=(28, 28, 1)):
        super(KeyEncoder, self).__init__()
        self.input_shape = input_shape
        self.latent_dim_key = latent_dim_key
        self.num_classes = num_classes

        # Fully connected layers (NN layers)
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(torch.prod(torch.tensor(self.input_shape))) + 
                      num_classes, 500),  # First NN layer
            nn.ReLU()
        )

        # Fully connected layers (NN layers)
        self.fc2 = nn.Sequential(
            nn.Linear(500, 500),  # Second NN layer
            nn.ReLU(),
            nn.Linear(500, latent_dim_key * 2)  # Output layer
        )

    def forward(self, x, labels):
        """
        Forward pass through the key_encoder.

        Args:
            x (Tensor): Input tensor with shape [batch_size, *input_shape].
            labels (Tensor): Conditional labels with shape [batch_size, num_classes].

        Returns:
            Latent space representation for the keys input.
        """
        # Flatten the input and concatenate with labels
        x = torch.cat([x.view(x.size(0), -1), labels], dim=1)

        # Pass through the first fully connected layer
        x = self.fc1(x)

        # Pass through the second fully connected layer
        x = self.fc2(x)

        return x.view(-1, self.latent_dim_key * 2)


class Decoder(nn.Module):
    """
    TODO: Why should I use `torch.prod` when calculating the size of the last layer???
    It can very well be `int(input_shape)`
    """
    def __init__(self, latent_dim_x=8, latent_dim_key=3, num_classes=2, input_shape=(28, 28, 1)):
        super(Decoder, self).__init__()
        self.input_shape = input_shape
        self.latent_dim_x = latent_dim_x
        self.latent_dim_key = latent_dim_key
        self.num_classes = num_classes

        # Fully connected layers (NN layers)
        self.fc1 = nn.Sequential(
            nn.Linear(latent_dim_x + latent_dim_key + num_classes, 800),  # First NN layer
            nn.ReLU()
        )

        # CNN layers
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=4, dilation=1),
            nn.MaxPool1d(kernel_size=4),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, dilation=2),
            nn.MaxPool1d(kernel_size=4),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=4, dilation=2),
            nn.MaxPool1d(kernel_size=4),
            nn.ReLU(),
        )

        # Fully connected layers (NN layers after CNN)
        self.fc2 = nn.Sequential(
            nn.Linear(64 * self._calculate_cnn_output_size(800), 800),  # Second NN layer
            nn.ReLU(),
            nn.Linear(800, int(torch.prod(torch.tensor(input_shape)))),  # Output layer
        )

    def _calculate_cnn_output_size(self, input_length):
        """
        Calculates the output size of the CNN layers after applying convolutions and pooling.

        Args:
            input_length (int): The input sequence length to the CNN layers.

        Returns:
            int: The size of the flattened CNN output.
        """
        # First CNN layer
        sequence_length = input_length
        sequence_length = ((sequence_length - (4 - 1) - 1) // 1 + 1)  # Conv1
        sequence_length = (sequence_length - 4) // 4 + 1  # Pool1

        # Second CNN layer
        sequence_length = ((sequence_length - (4 - 1) * 2 - 1) // 1 + 1)  # Conv2 (dilation=2)
        sequence_length = (sequence_length - 4) // 4 + 1  # Pool2

        # Third CNN layer
        sequence_length = ((sequence_length - (4 - 1) * 2 - 1) // 1 + 1)  # Conv3 (dilation=2)
        sequence_length = (sequence_length - 4) // 4 + 1  # Pool3

        return sequence_length

    def forward(self, z2, z2p, labels):
        """
        Forward pass through the decoder.

        Args:
            z2 (Tensor): Latent space representation for data.
            z2p (Tensor): Latent space representation for keys.
            labels (Tensor): Conditional labels.

        Returns:
            Tensor: Reconstructed input with the same shape as the original input.
        """

        # TODO: The input to the decoder here could
        # (z2+z2p)/2 which is concatenated with the labels.

        # Concatenate latent variables and labels along the feature dimension
        z = torch.cat([z2, z2p, labels], dim=1)
        logging.debug(z.shape)

        # Pass through the first fully connected layer
        x = self.fc1(z)

        # Reshape for CNN layers
        x = x.view(x.size(0), 1, -1)  # Reshape to [batch_size, channels, sequence_length]
        logging.debug(x.shape)

        # Pass through CNN layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Flatten and pass through the second fully connected layer
        logging.debug(x.shape)
        x = x.view(x.size(0), -1)
        logging.debug(x.shape)
        x = self.fc2(x)

        # Reshape to match the input shape
        return x.view(-1, *self.input_shape)


def check_for_nan_inf(tensor, name):
    if torch.isnan(tensor).any():
        logging.info(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        logging.info(f"Inf detected in {name}")


class CVAE(nn.Module):
    """
    Conditional Variational Autoencoder (CVAE) implementation based on the 
    paper: https://doi.org/10.1103/PhysRevD.103.124051

    This model encodes and decodes data conditioned on labels, with two 
    separate latent spaces for the data and the keys. It includes encoders 
    for both the data and the keys, as well as label-conditioned encoders. 
    The decoder reconstructs the input data from the latent representations 
    and labels.

    Attributes:
    -----------
    input_shape : tuple
        Shape of the input data.
    num_classes : int
        Number of classes for the conditional labels.
    latent_dim_x : int
        Dimension of the latent space for the data. Equal to 8.
    latent_dim_key : int
        Dimension of the latent space for the keys. Equal to 3.
    decoder : nn.Sequential
        Decoder network for reconstructing the input.

    Methods:
    --------
    x_encoder():
        Builds the encoder for the input data (E2 in the paper).
    key_encoder():
        Builds the encoder for the keys (E2prime in the paper).
    label_cond_for_x():
        Builds the label-conditioned encoder for the data (E1 in the paper).
    label_cond_for_key():
        Builds the label-conditioned encoder for the keys (E1prime in the paper).
    build_decoder():
        Builds the decoder network.
    encode_x(x, labels):
        Encodes the input data and labels into latent space z2.
    encode_key(keys, labels):
        Encodes the keys and labels into latent space z2prime.
    encode_label_for_x(labels):
        Encodes the labels into latent space z1 for the data.
    encode_label_for_key(labels):
        Encodes the labels into latent space z1prime for the keys.
    decode(z2, z1a, z2a, labels):
        Decodes the latent representations and labels to reconstruct the input.
    reparameterize(z_mean, z_log_var):
        Performs the reparameterization trick to sample from the latent space.
    latent_loss(z1, z2):
        Computes the latent loss between two latent representations.
    forward(x, labels):
        Forward pass through the CVAE, encoding and decoding the input.
    loss_function(x, x_recon, z_mean, z_log_var):
        Computes the total loss, including reconstruction and KL divergence.
    """
    def __init__(self, input_shape, num_classes, key_shape, \
                 labels_mean=None, labels_std=None, paramsnorm=False, \
                 latent_dim_x=8, latent_dim_key=3, MODEL_CONFIG=None):
        super(CVAE, self).__init__()

        # Override hyperparameters with MODEL_CONFIG values if provided
        # This allows for flexible model configuration while maintaining default values.
        if MODEL_CONFIG is not None:
            latent_dim_x = MODEL_CONFIG.get('latent_dim_x', latent_dim_x)
            latent_dim_key = MODEL_CONFIG.get('latent_dim_key', latent_dim_key)
            labels_mean = MODEL_CONFIG.get('labels_mean', labels_mean)
            labels_std = MODEL_CONFIG.get('labels_std', labels_std)
            num_classes = MODEL_CONFIG.get('num_classes', num_classes)

        # If MODEL_CONFIG is provided, it should contain all necessary hyperparameters.
        # If not provided, the default values will be used.
        self.latent_dim_x = latent_dim_x  # Dimension of the latent space
        self.latent_dim_key = latent_dim_key  # Dimension of the latent space
        self.input_shape = input_shape  # shape of strain array
        self.num_classes = num_classes  # shape of labels
        self.key_shape = key_shape      # shape of mean/var array

        # This works regardless of whether MODEL_CONFIG is provided or not, 
        # because if MODEL_CONFIG is not provided, the default values will be used.
        if paramsnorm:
            logging.info("Input parameter normalization is ENABLED. \
                The model will normalize the input parameters.")
            if labels_mean is None or labels_std is None:
                raise ValueError("labels_mean and labels_std must be provided when paramsnorm is True.")
            if not isinstance(labels_mean, torch.Tensor):
                labels_mean = torch.tensor(labels_mean, dtype=torch.float64)
            if not isinstance(labels_std, torch.Tensor):
                labels_std = torch.tensor(labels_std, dtype=torch.float64)
            self.register_buffer('labels_mean', labels_mean)
            self.register_buffer('labels_std', labels_std)
        elif labels_mean is not None or labels_std is not None:
            self.register_buffer('labels_mean', labels_mean)
            self.register_buffer('labels_std', labels_std)
            logging.warning("labels_mean and labels_std are provided but paramsnorm is False. \
                This means that labels_mean and labels_std were provided in MODEL_CONFIG. We will use them!")
        else:
            logging.info("Input parameter normalization is NOT enabled. \
                The model will use the raw labels without normalization.")

        # E2 in Fig 11 of the paper
        # self.x_encoder = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(int(torch.prod(torch.tensor(self.input_shape))) + 
        #               self.num_classes, 500),
        #     nn.ReLU(),

        #     # First CNN layer
        #     nn.Conv1d(in_channels=500, out_channels=5, kernel_size=16, dilation=1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=4),

        #     # Second CNN layer
        #     nn.Conv1d(in_channels=5, out_channels=15, kernel_size=16, dilation=1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=4),

        #     # Flatten the output of CNN layers
        #     nn.Flatten(),

        #     # Second Fully Connected (NN) layer
        #     nn.Linear(500, 500),
        #     nn.ReLU(),

        #     # Third Fully Connected (NN) layer
        #     nn.Linear(500, latent_dim_x * 2)  # Output z_mean and z_log_var
        # )

        # E2prime in Fig 11 of the paper
        # Fully connected layers (NN layers)
        self.key_encoder = nn.Sequential(
            nn.Flatten(),  # Flatten the input to a 1D vector
            nn.Linear(int(torch.prod(torch.tensor(self.key_shape))) + self.num_classes, 400),  # First NN layer
            nn.ReLU(),
            nn.Linear(400, 400),  # Second NN layer
            nn.ReLU(),
            nn.Linear(400, latent_dim_key * 2)  # Third NN layer (outputs z_mean and z_log_var)
        )
    
        # E1 in Fig 11 of the paper
        self.label_cond_for_x = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_classes, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, self.latent_dim_x * 2)
        )
    
        # E1prime in Fig 11 of the paper
        self.label_cond_for_key = nn.Sequential(
            nn.Linear(self.num_classes, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            # outputs are the mean and variance of the
            # latent space representation
            nn.Linear(500, self.latent_dim_key * 2)
        )

        self.x_encoder = XEncoder(latent_dim_x=self.latent_dim_x,
                               num_classes=self.num_classes,
                               input_shape=self.input_shape)
        self.decoder = Decoder(latent_dim_x=self.latent_dim_x,
                               latent_dim_key=self.latent_dim_key,
                               num_classes=self.num_classes,
                               input_shape=self.input_shape)
                               
    def __call__(self, x, labels, keys):
        """
        Overrides the __call__ method to directly call the forward method.
        
        Args:
            x (Tensor): Input data.
            labels (Tensor): Conditional labels.
            keys (Tensor): Key data.

        Returns:
            Output of the forward method.

        NOTE: Input labels normalization is only performed if `paramsnorm` is set to True 
        during initialization and `labels_mean` and `labels_std` are provided. 
        If `paramsnorm` is False, the labels will be used as they are without normalization.
        """
        # print('__call__')
        logging.debug(f'__call__ with x.shape={x.shape}, labels.shape={labels.shape}, keys.shape={keys.shape}')
        if hasattr(self, 'labels_mean') and hasattr(self, 'labels_std'):
            labels = self.normalize_labels(labels)
        return self.forward(x, labels, keys)

    def encode_x(self, x, labels):
        # Outputs `z2` from Fig 11 of the paper.
        # Flatten the input and concatenate with labels
        #x = torch.cat([x.view(x.size(0), -1), labels], dim=1)
        h = self.x_encoder(x, labels)
        z2_mean, z2_log_var = h.chunk(2, dim=1)
        # logging.debug(z2_log_var.shape)
        return z2_mean, z2_log_var
    
    def encode_key(self, keys, labels):
        # Outputs `z2prime` from Fig 11 of the paper.
        x = torch.cat([keys.view(keys.size(0), -1), labels], dim=1)
        # Since the key encoder doesn7t have any CNN layers, we can
        # directly concatenate the input and labels at the input pass.
        h = self.key_encoder(x)
        z2p_mean, z2p_log_var = h.chunk(2, dim=1)
        # logging.debug(z2p_mean, z2p_log_var)
        return z2p_mean, z2p_log_var

    def normalize_labels(self, labels, batchwise=False):
        """
        Normalize labels as: (label - mean) / std, where mean and std are calculated
        batch wise. The labels won't necessarily lie between [0,1]
        Uses the global mean and std calculated from the training data to ensure consistency 
        between training and inference.

        NOTE: Should never use batchwise normalization for the input parameter
        labels, because the mean and std will be different for each batch and thus 
        the model won't learn anything meaningful, although the training loss will
        decrease. During inference, the model will try to predict outputs based on
        the normalized labels specific 'to current batch in test set' and thus will,
        fail miserably in predicting correct outputs. The outputs will mostly resemble
        random noise-like curves, with slight twists at the merger stage.

        Arguments:
            labels (Tensor): The labels to be normalized.
            batchwise (bool): Whether to calculate mean and std for each batch or use global mean and std.
        """
        if batchwise:
            logging.warning("Batchwise normalization is not recommended for labels as it can lead " \
            "to inconsistent training and inference. Consider using global mean and std for normalization.")
            batch_mean = labels.mean(dim=0, keepdim=True)
            batch_std = labels.std(dim=0, keepdim=True) + 1e-8  # Add small value to avoid division by zero
            return (labels - batch_mean) / batch_std
        logging.debug(f"Normalizing labels with global mean: {self.labels_mean}, global std: {self.labels_std}")
        return (labels - self.labels_mean) / self.labels_std
    
    def encode_label_for_x(self, labels):
        # Outputs `z1` from Fig 11 of the paper.
        # print(labels)
        h = self.label_cond_for_x(labels)
        z1_mean, z1_log_var = h.chunk(2, dim=1)
        return z1_mean, z1_log_var
    
    def encode_label_for_key(self, labels):
        # Outputs `z1prime` from Fig 11 of the paper.
        h = self.label_cond_for_key(labels)
        z1p_mean, z1p_log_var = h.chunk(2, dim=1)
        # logging.debug(z1p_mean, z1p_log_var)
        return z1p_mean, z1p_log_var

    def decode(self, z2, z2p, labels):
        """
        Since this is the 2C2E1D model, the decoder will take the keys 
        (means) and the normalized strains, and then denormalize the 
        strains. I think this would be the correct way to do it.
        The labels are concatenated with the latent space outputs. 
        I think so!?

        The inputs to the Decoder part of the network are the latent
        space representation of the strains and the keys and labels
        for the data. These are then combined together and result in
        some decoded facsimile of the target, using which we can check
        the reconstruction loss and try to minimize it!
        ~~~
        Decodes the latent space representation (z2, z2p) along with labels
        to reconstruct the input. The latent variables and labels are concatenated
        and passed through the decoder to generate the reconstructed output.

        Args:
            z2 (Tensor): Latent space representation (e.g., strains).
            z2p (Tensor): Normalized latent space representation (e.g., keys).
            labels (Tensor): Labels associated with the data.

        Returns:
            Tensor: Reconstructed input with the same shape as the original input.
        """
        assert z2.size(0) == z2p.size(0) == labels.size(0) # Batch sizes must match

        # Pass the concatenated tensor through the decoder
        x_recon = self.decoder(z2, z2p, labels)
        
        # Reshape the output to match the input shape
        return x_recon.view(-1, *self.input_shape)

    def reparameterize(self, z_mean, z_log_var):
        # This is the variational part of the VAE
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    # def latent_loss(self, z1, z2):
    #     # TODO: Check if this is the correct loss function !
    #     # Latent loss is MSE loss for Auto-Encoder, not VAE!
    #     mse = F.mse_loss(z1, z2, reduction='none')
    #     return mse.sum(1)

    def latent_loss(self, z_mean, z_log_var):
        """
        Computes the KL divergence between the approximate posterior q(z|x)
        and the prior p(z) (assumed to be a standard Gaussian N(0, I)).

        Parameters:
        -----------
        z_mean : torch.Tensor
            Mean of the latent space distribution.
        z_log_var : torch.Tensor
            Log variance of the latent space distribution.

        Returns:
        --------
        kl_loss : torch.Tensor
            KL divergence loss for the latent space.
        """
        # Clamp log variance to avoid numerical instability
        z_log_var = torch.clamp(z_log_var, min=-10, max=10)
        # print(z_log_var)
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=1)
        return kl_loss.mean()  # Average over the batch
    
    def latent_loss_between_encoders(self, z1_mean, z1_log_var, z2_mean, z2_log_var):
        """
        Computes the KL divergence between two Gaussian distributions
        generated by the two encoders.

        Parameters:
        -----------
        z1_mean : torch.Tensor
            Mean of the first Gaussian distribution (from the first encoder).
        z1_log_var : torch.Tensor
            Log variance of the first Gaussian distribution (from the first encoder).
        z2_mean : torch.Tensor
            Mean of the second Gaussian distribution (from the second encoder).
        z2_log_var : torch.Tensor
            Log variance of the second Gaussian distribution (from the second encoder).

        Returns:
        --------
        kl_loss : torch.Tensor
            KL divergence loss between the two Gaussian distributions.
        """
        # Variances
        sigma1_sq = torch.exp(z1_log_var)
        sigma2_sq = torch.exp(z2_log_var)

        # KL divergence
        kl_loss = 0.5 * torch.sum(
            (sigma1_sq / sigma2_sq) + ((z2_mean - z1_mean).pow(2) / sigma2_sq) - 1 + (z2_log_var - z1_log_var),
            dim=1
        )
        return kl_loss.mean()  # Average over the batch

    def forward(self, x, labels, keys):
        """
        The input to the Decoder part of the model are the reparametrized
        latent space representations. Reparametrization is necessary bcuz
        it introduces some random error in the distribution and enables
        random sampling from the latent space distribution, which we assume
        is as close to a Gaussian as possible.
        Decoder doesn't need `z1` and `z1p` bcuz these are simply label
        conditioners and are used to calculate the latent space errors. Thus,
        no parametrization is necessary for these.
        """
        logging.debug(keys.shape)
        logging.debug(f'labels.shape={labels.shape}')
        # print(keys)
        z1_mean, z1_log_var = self.encode_label_for_x(labels)
        z2_mean, z2_log_var = self.encode_x(x, labels)
        check_for_nan_inf(z1_mean, 'z1_mean')
        check_for_nan_inf(z2_mean, 'z2_mean')
        check_for_nan_inf(z1_log_var, 'z1_log_var')
        check_for_nan_inf(z2_log_var, 'z2_log_var')
        # print("z1_mean:", z1_mean)
        # print("z1_log_var:", z1_log_var)
        # print("z2_mean:", z2_mean)
        # print("z2_log_var:", z2_log_var)
        z1p_mean, z1p_log_var = self.encode_label_for_key(labels)
        z2p_mean, z2p_log_var = self.encode_key(keys, labels)
        check_for_nan_inf(z1p_mean, 'z1p_mean')
        check_for_nan_inf(z2p_mean, 'z2p_mean')
        check_for_nan_inf(z1p_log_var, 'z1p_log_var')
        check_for_nan_inf(z2p_log_var, 'z2p_log_var')
        # print("z1p_mean:", z1p_mean)
        # print("z1p_log_var:", z1p_log_var)
        # print("z2p_mean:", z2p_mean)
        # print("z2p_log_var:", z2p_log_var)
        z2 = self.reparameterize(z2_mean, z2_log_var)
        z2p = self.reparameterize(z2p_mean, z2p_log_var)
        # TODO: Should be inputting label embeddings, instead of raw labels!
        x_recon = self.decode(z2, z2p, labels)
        zvars = [z1_mean, z1_log_var, z2_mean, z2_log_var, \
                 z1p_mean, z1p_log_var, z2p_mean, z2p_log_var]
        return (x_recon, zvars)

    def loss_function(self, x, x_recon, zvars):
        """
        Computes the total loss for the CVAE, including reconstruction loss,
        KL divergence for latent spaces, and latent loss between encoders.

        Parameters:
        -----------
        x : torch.Tensor
            Original input data.
        x_recon : torch.Tensor
            Reconstructed input data.
        z1_mean, z1_log_var : torch.Tensor
            Mean and log variance for latent space z1.
        z2_mean, z2_log_var : torch.Tensor
            Mean and log variance for latent space z2.
        z1p_mean, z1p_log_var : torch.Tensor
            Mean and log variance for latent space z1p.
        z2p_mean, z2p_log_var : torch.Tensor
            Mean and log variance for latent space z2p.

        Returns:
        --------
        total_loss : torch.Tensor
            Total loss combining reconstruction and latent losses.
        """
        z1_mean, z1_log_var, z2_mean, z2_log_var, \
            z1p_mean, z1p_log_var, z2p_mean, z2p_log_var = zvars
        logging.debug(f'z1_mean={z1_mean}, z1_log_var={z1_log_var}, z2_mean={z2_mean}, z2_log_var={z2_log_var}, \
            z1p_mean={z1p_mean}, z1p_log_var={z1p_log_var}, z2p_mean={z2p_mean}, z2p_log_var={z2p_log_var}')

        # Reconstruction loss (e.g., Binary Cross-Entropy or MSE)
        # TODO: What is the `reduction` thing doing here?
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        logging.debug(f"Reconstruction Loss: {recon_loss.item()}")

        # KL divergence for each latent space
        kl_loss_z1 = self.latent_loss(z1_mean, z1_log_var)
        kl_loss_z2 = self.latent_loss(z2_mean, z2_log_var)
        kl_loss_z1p = self.latent_loss(z1p_mean, z1p_log_var)
        kl_loss_z2p = self.latent_loss(z2p_mean, z2p_log_var)
        # Print KL divergence losses for debugging
        logging.debug(f"KL Loss z1: {kl_loss_z1.item()}, KL Loss z2: {kl_loss_z2.item()}, "
            f"KL Loss z1p: {kl_loss_z1p.item()}, KL Loss z2p: {kl_loss_z2p.item()}")

        # Latent loss between encoders
        ll1 = self.latent_loss_between_encoders(z1_mean, z1_log_var, z2_mean, z2_log_var)
        ll2 = self.latent_loss_between_encoders(z1p_mean, z1p_log_var, z2p_mean, z2p_log_var)

        # Total loss
        beta = 0.1   # Weighting factor for KL divergence
        kl_loss = kl_loss_z1 + kl_loss_z2 + kl_loss_z1p + kl_loss_z2p + ll1 + ll2
        total_loss = recon_loss + beta * kl_loss
        return (total_loss, recon_loss, kl_loss)

    def mismatch_loss_func(self, x, x_recon, zvars, strains, keys, attr):
        """
        Computes the mismatch loss between the reconstructed output and the keys.

        Parameters:
        -----------
        x : torch.Tensor
            Original input data.
        x_recon : torch.Tensor
            Reconstructed input data.
        zvars : list of torch.Tensor
            List of latent variable means and log variances.
        keys : torch.Tensor
            Normalization keys for the input amplitude and frequency data.

        Returns:
        --------
        total_loss : torch.Tensor
            Total loss combining reconstruction, latent losses, and mismatch loss.
        """
        z1_mean, z1_log_var, z2_mean, z2_log_var, \
            z1p_mean, z1p_log_var, z2p_mean, z2p_log_var = zvars
        logging.debug(f'z1_mean={z1_mean}, z1_log_var={z1_log_var}, z2_mean={z2_mean}, z2_log_var={z2_log_var}, \
            z1p_mean={z1p_mean}, z1p_log_var={z1p_log_var}, z2p_mean={z2p_mean}, z2p_log_var={z2p_log_var}')

        # Reconstruction loss (e.g., Binary Cross-Entropy or MSE)
        # TODO: What is the `reduction` thing doing here?
        # NOTE: This is good to have since we also wnat to have the
        # output amplitude and frequency series to have proper inspiral
        # stage reconstructions. The mismatch loss on the other hand, will
        # calculate the mismatch between the reconstructed and the original waveforms
        # thereby making sure that the phase evolution and merger time freq
        # changes are correctly captured by the model.
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')

        # KL divergence for each latent space
        kl_loss_z1 = self.latent_loss(z1_mean, z1_log_var)
        kl_loss_z2 = self.latent_loss(z2_mean, z2_log_var)
        kl_loss_z1p = self.latent_loss(z1p_mean, z1p_log_var)
        kl_loss_z2p = self.latent_loss(z2p_mean, z2p_log_var)
        # Print KL divergence losses for debugging
        logging.info(f"KL Loss z1: {kl_loss_z1.item()}, KL Loss z2: {kl_loss_z2.item()}, "
            f"KL Loss z1p: {kl_loss_z1p.item()}, KL Loss z2p: {kl_loss_z2p.item()}")

        # Latent loss between encoders
        ll1 = self.latent_loss_between_encoders(z1_mean, z1_log_var, z2_mean, z2_log_var)
        ll2 = self.latent_loss_between_encoders(z1p_mean, z1p_log_var, z2p_mean, z2p_log_var)

        # Total loss
        beta = 0.1   # Weighting factor for KL divergence
        kl_loss = kl_loss_z1 + kl_loss_z2 + kl_loss_z1p + kl_loss_z2p + ll1 + ll2


        # # -- Calculate the total mismatch loss for the batch
        # mmloss = 0.0
        # for i in range(x.size(0)):
        #     delta_t = attr['delta_t'][i]
        #     f_lower = attr['f_lower'][i]
        #     # -- Get the reconstructed and original waveforms for the i-th sample
        #     x_recon_i = x_recon[i].cpu().detach().numpy()
        #     x_i = x[i].cpu().detach().numpy()
        #     amp_recon, freq_recon = x_recon_i[0], x_recon_i[1]
        #     amp_orig, freq_orig = x_i[0], x_i[1]
        #     # -- denormalize amp and freq using the keys
        #     key = keys[i].reshape([2,2]).cpu().detach().numpy()
        #     amp_mean, amp_std = key[0][0], key[0][1]
        #     freq_mean, freq_std = key[1][0], key[1][1]
        #     amp_recon = (amp_recon * amp_std) + amp_mean
        #     freq_recon = (freq_recon * freq_std) + freq_mean
        #     amp_orig = (amp_orig * amp_std) + amp_mean
        #     freq_orig = (freq_orig * freq_std) + freq_mean
        #     # -- remove first dummy element from the frequency series
        #     freq_recon = freq_recon[1:]
        #     freq_orig = freq_orig[1:]
        #     logging.debug(f'Shapes: amp_recon={amp_recon.shape}, freq_recon={freq_recon.shape}, amp_orig={amp_orig.shape}, freq_orig={freq_orig.shape}')
        #     # -- calculate start phase / reference phase
        #     hp_hdf = strains[i][0].cpu().detach().numpy()
        #     hc_hdf = strains[i][1].cpu().detach().numpy()
        #     phase_hdf = np.unwrap(np.arctan2(hc_hdf, hp_hdf))
        #     # -- Calculate the mismatch loss for the i-th sample
        #     hp_recon, hc_recon = polarizations_from_ampfreq(amp_recon, freq_recon, theta0=phase_hdf[0])
        #     hp_orig, hc_orig = polarizations_from_ampfreq(amp_orig, freq_orig, theta0=phase_hdf[0])
        #     mmloss_hp_i = calc_polarization_mismatch(hp_recon, hp_orig, delta_t=delta_t, f_lower=f_lower)
        #     mmloss_hc_i = calc_polarization_mismatch(hc_recon, hc_orig, delta_t=delta_t, f_lower=f_lower)
        #     logging.debug(f'Mismatch losses for sample {i}: mmloss_hp_i={mmloss_hp_i}, mmloss_hc_i={mmloss_hc_i}')
        #     mmloss += (mmloss_hp_i + mmloss_hc_i) / 2.0

        # -- Calculate the total mismatch loss for the batch (vectorized)
        amp_recon = x_recon[:, 0].cpu().detach().numpy()
        freq_recon = x_recon[:, 1].cpu().detach().numpy()
        amp_orig = x[:, 0].cpu().detach().numpy()
        freq_orig = x[:, 1].cpu().detach().numpy()
        
        # -- Denormalize using keys (vectorized)
        keys_reshaped = keys.reshape(-1, 2, 2).cpu().detach().numpy()
        amp_mean, amp_std = keys_reshaped[:, 0, 0], keys_reshaped[:, 0, 1]
        freq_mean, freq_std = keys_reshaped[:, 1, 0], keys_reshaped[:, 1, 1]
        
        amp_recon = (amp_recon * amp_std[:, np.newaxis]) + amp_mean[:, np.newaxis]
        freq_recon = (freq_recon * freq_std[:, np.newaxis]) + freq_mean[:, np.newaxis]
        amp_orig = (amp_orig * amp_std[:, np.newaxis]) + amp_mean[:, np.newaxis]
        freq_orig = (freq_orig * freq_std[:, np.newaxis]) + freq_mean[:, np.newaxis]
        
        # -- Remove first dummy element from frequency series
        freq_recon = freq_recon[:, 1:]
        freq_orig = freq_orig[:, 1:]
        
        # -- Calculate phase (vectorized)
        hp_hdf = strains[:, 0].cpu().detach().numpy()
        hc_hdf = strains[:, 1].cpu().detach().numpy()
        phase_hdf = np.unwrap(np.arctan2(hc_hdf, hp_hdf), axis=1)
        
        # -- Calculate mismatch loss (vectorized)
        mmloss = 0.0
        for i in range(x.size(0)):
            hp_recon, hc_recon = polarizations_from_ampfreq(amp_recon[i], freq_recon[i], theta0=phase_hdf[i, 0])
            hp_orig, hc_orig = polarizations_from_ampfreq(amp_orig[i], freq_orig[i], theta0=phase_hdf[i, 0])
            mmloss_hp_i = calc_polarization_mismatch(hp_recon, hp_orig, delta_t=attr['delta_t'][i], f_lower=attr['f_lower'][i])
            mmloss_hc_i = calc_polarization_mismatch(hc_recon, hc_orig, delta_t=attr['delta_t'][i], f_lower=attr['f_lower'][i])
            mmloss += (mmloss_hp_i + mmloss_hc_i) / 2.0
        
        logging.info(f'Total mismatch loss for the batch: {mmloss}')
        total_loss = recon_loss + beta * kl_loss + mmloss
        return (total_loss, recon_loss, kl_loss, mmloss)

    def mismatch_nokl_loss_func(self, x, x_recon, zvars, strains, keys, attr):
        """
        Computes the mismatch loss between the reconstructed output and the keys.

        Parameters:
        -----------
        x : torch.Tensor
            Original input data.
        x_recon : torch.Tensor
            Reconstructed input data.
        zvars : list of torch.Tensor
            List of latent variable means and log variances.
        keys : torch.Tensor
            Normalization keys for the input amplitude and frequency data.

        Returns:
        --------
        total_loss : torch.Tensor
            Total loss combining reconstruction, latent losses, and mismatch loss.
        """
        z1_mean, z1_log_var, z2_mean, z2_log_var, \
            z1p_mean, z1p_log_var, z2p_mean, z2p_log_var = zvars
        logging.debug(f'z1_mean={z1_mean}, z1_log_var={z1_log_var}, z2_mean={z2_mean}, z2_log_var={z2_log_var}, \
            z1p_mean={z1p_mean}, z1p_log_var={z1p_log_var}, z2p_mean={z2p_mean}, z2p_log_var={z2p_log_var}')

        # Reconstruction loss (e.g., Binary Cross-Entropy or MSE)
        # TODO: What is the `reduction` thing doing here?
        # NOTE: This is good to have since we also wnat to have the
        # output amplitude and frequency series to have proper inspiral
        # stage reconstructions. The mismatch loss on the other hand, will
        # calculate the mismatch between the reconstructed and the original waveforms
        # thereby making sure that the phase evolution and merger time freq
        # changes are correctly captured by the model.
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')

        # -- Calculate the total mismatch loss for the batch (vectorized)
        amp_recon = x_recon[:, 0].cpu().detach().numpy()
        freq_recon = x_recon[:, 1].cpu().detach().numpy()
        amp_orig = x[:, 0].cpu().detach().numpy()
        freq_orig = x[:, 1].cpu().detach().numpy()
        
        # -- Denormalize using keys (vectorized)
        keys_reshaped = keys.reshape(-1, 2, 2).cpu().detach().numpy()
        amp_mean, amp_std = keys_reshaped[:, 0, 0], keys_reshaped[:, 0, 1]
        freq_mean, freq_std = keys_reshaped[:, 1, 0], keys_reshaped[:, 1, 1]
        
        amp_recon = (amp_recon * amp_std[:, np.newaxis]) + amp_mean[:, np.newaxis]
        freq_recon = (freq_recon * freq_std[:, np.newaxis]) + freq_mean[:, np.newaxis]
        amp_orig = (amp_orig * amp_std[:, np.newaxis]) + amp_mean[:, np.newaxis]
        freq_orig = (freq_orig * freq_std[:, np.newaxis]) + freq_mean[:, np.newaxis]
        
        # -- Remove first dummy element from frequency series
        freq_recon = freq_recon[:, 1:]
        freq_orig = freq_orig[:, 1:]
        
        # -- Calculate phase (vectorized)
        hp_hdf = strains[:, 0].cpu().detach().numpy()
        hc_hdf = strains[:, 1].cpu().detach().numpy()
        phase_hdf = np.unwrap(np.arctan2(hc_hdf, hp_hdf), axis=1)
        
        # -- Calculate mismatch loss (vectorized)
        mmloss = 0.0
        for i in range(x.size(0)):
            hp_recon, hc_recon = polarizations_from_ampfreq(amp_recon[i], freq_recon[i], theta0=phase_hdf[i, 0])
            hp_orig, hc_orig = polarizations_from_ampfreq(amp_orig[i], freq_orig[i], theta0=phase_hdf[i, 0])
            mmloss_hp_i = calc_polarization_mismatch(hp_recon, hp_orig, delta_t=attr['delta_t'][i], f_lower=attr['f_lower'][i])
            mmloss_hc_i = calc_polarization_mismatch(hc_recon, hc_orig, delta_t=attr['delta_t'][i], f_lower=attr['f_lower'][i])
            mmloss += (mmloss_hp_i + mmloss_hc_i) / 2.0
        
        logging.info(f'Total mismatch loss for the batch: {mmloss}')
        total_loss = recon_loss + mmloss
        return (total_loss, recon_loss, mmloss)
    
    def convert_output(self, output) -> np.ndarray:
        """
        Converts the output of the decoder (Amplitude and Frequency series) to the polarizations (hp and hc).
        This function is used in the generation step to convert the generated Amplitude and Frequency series to the polarizations, 
        which are the actual waveforms that we want to generate.

        NOTE: We assume the starting phase to be zero here! Which is not always correct and may result in
        incorrect resultant waveform! This difficulty is resolved in the FlexCAEPhase model, where we directly output 
        the phase instead of the frequency, and thus we can directly convert the [amp, phase] output to the polarizations 
        without assuming any starting phase!
        TODO: Remove dependency to detach the outputs from the device and numpy operations.
        TODO: All of this calculation should be done on a GPU.

        Returns:
        --------
        np.ndarray: Array of shape (batch_size, 2, sequence_length) containing the hp and hc polarizations for each sample in the batch.
        """
        print(f"Output shape: {output.shape}")
        amp = output[:, 0].cpu().detach().numpy()
        freq = output[:, 1].cpu().detach().numpy()

        # NOTE: When loading the data using "CustomDataset", I append a dummy element at the start of the frequency array, to make it the same length as the amplitude array (by definition it will be one element less in length), just the output of the trained model contains an extra element at the start which we can remove.
        # -- remove the first dummy element from the frequency array
        freq = freq[:, 1:]

        hphc = np.zeros((output.size(0), 2, amp.shape[1]))  # Initialize array to hold hp and hc
        for i in range(output.size(0)):
            hp, hc = polarizations_from_ampfreq(amp[i], freq[i], theta0=0.0)  # Assuming theta0=0 for generation
            hphc[i] = np.stack([hp, hc])
        return hphc

    def generate(self, labels=None) -> np.ndarray:
        """
        From a trained model, generate new output waveforms using only the
        conditional labels information, by sampling from the latent space and 
        passing through the decoder.

        Arguments:
        ---------    
        labels: torch.Tensor
            The conditional labels to generate waveforms for. Must be provided!

        Returns:
        --------
        hphc: np.ndarray
            The generated waveforms in the form of hp and hc polarizations, with shape 
            (batch_size, 2, sequence_length).
        """
        if labels is not None:
            # -- check dimensions of labels with MODEL_CONFIG['num_classes']
            assert labels.size(1) == self.num_classes, f"Labels dimension {labels.size(1)} does not match MODEL_CONFIG['num_classes'] {self.num_classes}!"
        else:
            raise ValueError("Labels must be provided for generation, since the model is conditional on the labels!")

        self.eval()  # Set model to evaluation mode
        logging.info("Model set to evaluation mode for generation.")

        with torch.no_grad():
            # Encode labels to get the mean and log variance of the latent space
            zy_mu, zy_logvar = self.encode_label_for_x(labels)
            zykey_mu, zykey_logvar = self.encode_label_for_key(labels)
            zy = self.reparameterize(zy_mu, zy_logvar)
            zykey = self.reparameterize(zykey_mu, zykey_logvar)
            # NOTE: In this original model, we do not have options to concatenate 
            generated_output = self.decode(zy, zykey, labels)

        hphc = self.convert_output(generated_output)
        return hphc


class CAE(CVAE):
    """
    Conditional Autoencoder (CAE) implementation that inherits from CVAE.
    This model is a simplified version of the CVAE, the latent space is not 
    regularized to follow a Gaussian distribution using the reparametrization
    trick. The CAE loss function still includes the KL divergence term to
    facilitate waveform generation beyond the discrete training set, but the 
    forward pass only takes in the mean of the latent space distribution given
    as an output of the encoder, and not the reparametrized latent variable. 
    This means that the model is now deterministic and not limited by the noise
    floor due to the reparametrization trick, but it can still generate waveforms that
    are not in the training set due to the KL divergence loss term, which we are
    minimizing in the loss function. Validation is also performed at the end of
    each epoch, which generalizes the model beyond the training set and ensures that 
    the model is not just memorizing the training data.
    The weightage for the KL divergence will only be 10%.

    Attributes:
    -----------
    Inherits all attributes from CVAE.

    Methods:
    --------
    forward(x, labels, keys):
        Overrides the forward method to remove the reparameterization step.
    """
    def forward(self, x, labels, keys):
        """
        Overrides the forward method to remove the reparameterization step.
        The latent space representations are directly taken as the mean outputs
        from the encoders without sampling, making the model deterministic.
        However, we still calculate the KL divergence loss in the loss function to encourage
        the latent space to follow a Gaussian distribution, which allows for generalization
        beyond the training set. The weightage for the KL divergence will only be 10%.
        """
        logging.debug(keys.shape)
        logging.debug(f'labels.shape={labels.shape}')
        # print(keys)
        z1_mean, z1_log_var = self.encode_label_for_x(labels)
        z2_mean, z2_log_var = self.encode_x(x, labels)
        check_for_nan_inf(z1_mean, 'z1_mean')
        check_for_nan_inf(z2_mean, 'z2_mean')
        check_for_nan_inf(z1_log_var, 'z1_log_var')
        check_for_nan_inf(z2_log_var, 'z2_log_var')
        # print("z1_mean:", z1_mean)
        # print("z1_log_var:", z1_log_var)
        # print("z2_mean:", z2_mean)
        # print("z2_log_var:", z2_log_var)
        z1p_mean, z1p_log_var = self.encode_label_for_key(labels)
        z2p_mean, z2p_log_var = self.encode_key(keys, labels)
        check_for_nan_inf(z1p_mean, 'z1p_mean')
        check_for_nan_inf(z2p_mean, 'z2p_mean')
        check_for_nan_inf(z1p_log_var, 'z1p_log_var')
        check_for_nan_inf(z2p_log_var, 'z2p_log_var')
        x_recon = self.decode(z2_mean, z2p_mean, labels)
        logging.debug(f'Encoded input: z2_mean={z2_mean}, z2p_mean={z2p_mean}')
        zvars = [z1_mean, z1_log_var, z2_mean, z2_log_var, \
                 z1p_mean, z1p_log_var, z2p_mean, z2p_log_var]
        return (x_recon, zvars)
    


# Example training loop
def train(model, data_loader, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for x, target, labels, keys, strains, attr in data_loader:
            optimizer.zero_grad()
            x_recon, zvars = model(x, labels, keys)
            loss, reconloss, klloss, mmloss = model.mismatch_loss_func(x, x_recon, zvars, keys, strains=strains, attr=attr)
            loss.backward()
            optimizer.step()
        logging.debug(f'Epoch {epoch + 1}, Loss: {loss.item()}, Recon Loss: {reconloss.item()}, KL Loss: {klloss.item()}, Mismatch Loss: {mmloss.item()}')

# Assuming `data_loader` is defined and provides batches of (data, labels)
# train(cvae, data_loader, optimizer)


if __name__=='__main__':

    import graphviz
    graphviz.set_jupyter_format('png')
    from torchview import draw_graph
    from datacvae import CustomDataset

    # Need to use `force` here to override default `logging` settings
    log_level = logging.INFO
    logging.basicConfig(format='%(levelname)s | %(asctime)s: %(message)s',
                                level=log_level, datefmt='%y-%m-%d %H:%M:%S',
                                force=True)

    logging.debug('Testing CVAE')

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

    cvae = CVAE(input_shape=(2, 8191), num_classes=2, key_shape=(2,2)).to(device)
    optimizer = torch.optim.Adam(cvae.parameters(), lr=1e-3)
    logging.debug('Model initialized')

    ds = CustomDataset(forwhat='train')
    batchsize = 50
    data_loader = torch.utils.data.DataLoader(ds, batch_size=batchsize, shuffle=True)
    logging.debug('Data loaded')

    logging.info('Visualizing the network!')
    timestamp = datetime.now().strftime('%Y%m%d')
    x, labels, keys = next(iter(data_loader))
    print(x.shape, labels.shape, keys.shape)
    cvae.train()
    # cvae(x.to(dtype=torch.float64), labels.to(dtype=torch.float64), keys.to(dtype=torch.float64)),
    #print( CVAE(input_shape=(2, 8191), num_classes=2, key_shape=(2,2))(x=x.to(dtype=torch.float64), labels=labels.to(dtype=torch.float64), keys=keys.to(dtype=torch.float64)),)
    # CVAE(input_shape=(2, 8191), num_classes=2, key_shape=(2,2)).to(device)(x=x.to(dtype=torch.float64), labels=labels.to(dtype=torch.float64), keys=keys.to(dtype=torch.float64))
    print(type(cvae))
    print(type(cvae(x=x.to(dtype=torch.float64), labels=labels.to(dtype=torch.float64), keys=keys.to(dtype=torch.float64))))
    modelgraph = draw_graph(cvae, input_data=(x.to(dtype=torch.float64), labels.to(dtype=torch.float64), keys.to(dtype=torch.float64)),
                        )
    modelgraph.visual_graph.render('cvae_model-'+timestamp, format='png', cleanup=True)

    # Example training loop
    # train(cvae, data_loader, optimizer, num_epochs=10)