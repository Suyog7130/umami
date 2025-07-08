import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal, kl_divergence

import matplotlib.pyplot as plt


from datetime import datetime


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
        logging.info(f'sequence_length={sequence_length}')
        # First CNN layer
        sequence_length = ((sequence_length - (16 - 1) - 1) // 1 + 1)  # Conv1
        sequence_length = (sequence_length - 4) // 4 + 1  # Pool1

        # Second CNN layer
        sequence_length = ((sequence_length - (16 - 1) - 1) // 1 + 1)  # Conv2
        sequence_length = (sequence_length - 4) // 4 + 1  # Pool2
        logging.info(f'sequence_length={sequence_length}')
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
                 latent_dim_x=8, latent_dim_key=3):
        super(CVAE, self).__init__()
        self.latent_dim_x = latent_dim_x  # Dimension of the latent space
        self.latent_dim_key = latent_dim_key  # Dimension of the latent space
        self.input_shape = input_shape  # shape of strain array
        self.num_classes = num_classes  # shape of labels
        self.key_shape = key_shape      # shape of mean/var array

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
        """
        print('__call__')
        return self.forward(x, labels, keys)

    def encode_x(self, x, labels):
        # Outputs `z2` from Fig 11 of the paper.
        # Flatten the input and concatenate with labels
        #x = torch.cat([x.view(x.size(0), -1), labels], dim=1)
        h = self.x_encoder(x, labels)
        z2_mean, z2_log_var = h.chunk(2, dim=1)
        logging.debug(z2_log_var.shape)
        return z2_mean, z2_log_var
    
    def encode_key(self, keys, labels):
        # Outputs `z2prime` from Fig 11 of the paper.
        x = torch.cat([keys.view(keys.size(0), -1), labels], dim=1)
        # Since the key encoder doesn7t have any CNN layers, we can
        # directly concatenate the input and labels at the input pass.
        h = self.key_encoder(x)
        z2p_mean, z2p_log_var = h.chunk(2, dim=1)
        logging.debug(z2p_mean, z2p_log_var)
        return z2p_mean, z2p_log_var
    
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
        logging.debug(z1p_mean, z1p_log_var)
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

        # Reconstruction loss (e.g., Binary Cross-Entropy or MSE)
        # TODO: What is the `reduction` thing doing here?
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
        total_loss = recon_loss + beta * kl_loss
        return (total_loss, recon_loss, kl_loss)
    


# Example training loop
def train(model, data_loader, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for x, labels, keys in data_loader:
            optimizer.zero_grad()
            x_recon, zvars = model(x, labels, keys)
            loss = model.loss_function(x, x_recon, zvars)
            loss.backward()
            optimizer.step()
        logging.debug(f'Epoch {epoch + 1}, Loss: {loss.item()}')

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
    # cvae(x.to(dtype=torch.float32), labels.to(dtype=torch.float32), keys.to(dtype=torch.float32)),
    #print( CVAE(input_shape=(2, 8191), num_classes=2, key_shape=(2,2))(x=x.to(dtype=torch.float32), labels=labels.to(dtype=torch.float32), keys=keys.to(dtype=torch.float32)),)
    # CVAE(input_shape=(2, 8191), num_classes=2, key_shape=(2,2)).to(device)(x=x.to(dtype=torch.float32), labels=labels.to(dtype=torch.float32), keys=keys.to(dtype=torch.float32))
    print(type(cvae))
    print(type(cvae(x=x.to(dtype=torch.float32), labels=labels.to(dtype=torch.float32), keys=keys.to(dtype=torch.float32))))
    modelgraph = draw_graph(cvae, input_data=(x.to(dtype=torch.float32), labels.to(dtype=torch.float32), keys.to(dtype=torch.float32)),
                        )
    modelgraph.visual_graph.render('cvae_model-'+timestamp, format='png', cleanup=True)

    # Example training loop
    # train(cvae, data_loader, optimizer, num_epochs=10)