"""
Flexible CVAE implementation using my class from `cvae.py` code.
Works for both CVAE and CAE configurations, with 2 encoders or 1 encoder.
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from datetime import datetime

from utils import polarizations_from_ampfreq, calc_polarization_mismatch

from typing import List, Sequence, Optional, Union, Callable



# ----------------------
# Small helper utilities
# ----------------------

def _as_list(x) -> List:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _pair_from_sizes(sizes: Sequence[int]):
    """
    Given a list of sizes, [a, b, c, ...], returns two lists:
    in_features = [a, b, c, ...] (all but last)
    out_features = [b, c, ...] (all but first)
    These will be used to construct the layers of the FC blocks,
    like: Linear(a, b) → Linear(b, c) → ... for the pre-FC and post-FC stages.
    The number of layers should be len(sizes) - 1, and the sizes should be compatible for chaining.
    """
    sizes = list(sizes)
    assert len(sizes) >= 2, "sizes must have len >= 2"
    in_features = sizes[:-1]  # all but last
    out_features = sizes[1:]  # all but first
    return in_features, out_features


def _make_activation(name: Optional[str]) -> nn.Module:
    if name is None:
        return nn.Identity()
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "elu":
        return nn.ELU()
    if name in ("leaky_relu", "lrelu", "leaky"):
        return nn.LeakyReLU(0.2)
    raise ValueError(f"Unknown activation: {name}")

class BaseCoder(nn.Module):
    """
    General base coder inherited by encoders/decoders.
    Contains atleast one pre-FC, CNN, and post-FC layer, with flexible sizes and counts for each stage.
    The number of each layer type can be more than 1, and the actual layers are built in the forward 
    pass based on the provided configuration,
    allowing for dynamic architectures. The configuration is designed to be flexible and can 
    specify either the full sizes list or just the in/out features for FC layers, and similarly 
    for CNN layers.
    """
    def __init__(self, **kwargs):
        super(BaseCoder, self).__init__()
        # Shapes & dims
        self.input_shape = kwargs.get('input_shape', None)      # e.g., (C, L) or (D,)
        self.latent_dim  = kwargs.get('latent_dim', None)
        self.num_classes = kwargs.get('num_classes', None)

        # Stage toggles
        self.has_pre_fc  = kwargs.get('has_pre_fc', True)
        self.has_cnn     = kwargs.get('has_cnn', True)
        self.has_post_fc = kwargs.get('has_post_fc', True)

        # Layer counts
        self.n_layers            = kwargs.get('n_layers', 1)
        self.n_layers_cnn        = kwargs.get('n_layers_cnn', self.n_layers)
        self.n_layers_fc         = kwargs.get('n_layers_fc', self.n_layers)
        self.n_layers_pre_fc     = kwargs.get('n_layers_pre_fc', self.n_layers_fc)
        self.n_layers_post_fc    = kwargs.get('n_layers_post_fc', self.n_layers_fc)

        # FC sizing
        self.pre_fc_in_features  = _as_list(kwargs.get('pre_fc_in_features', None))
        self.pre_fc_out_features = _as_list(kwargs.get('pre_fc_out_features', None))
        self.pre_fc_sizes        = _as_list(kwargs.get('pre_fc_sizes', None))  # optional: [in, ..., out]

        self.post_fc_in_features  = _as_list(kwargs.get('post_fc_in_features', None))
        self.post_fc_out_features = _as_list(kwargs.get('post_fc_out_features', None))
        self.post_fc_sizes        = _as_list(kwargs.get('post_fc_sizes', None))

        # CNN sizing
        self.cnn_in_channels   = _as_list(kwargs.get('cnn_in_channels', None))
        self.cnn_out_channels  = _as_list(kwargs.get('cnn_out_channels', None))
        self.cnn_kernel_size   = _as_list(kwargs.get('cnn_kernel_size', None))
        self.cnn_dilation      = _as_list(kwargs.get('cnn_dilation', 1))
        self.cnn_pool_ks       = _as_list(kwargs.get('cnn_pool_kernel_size', None))  # default: use kernel_size

        # Regularization & activations
        self.activation_name   = kwargs.get('activation', 'relu')
        self.last_activation   = kwargs.get('last_activation', None)  # None → Identity
        self.use_batchnorm     = kwargs.get('use_batchnorm', False)
        self.dropout_p         = kwargs.get('dropout_p', 0.0)
        self.use_dropout       = self.dropout_p is not None and self.dropout_p > 0.0

        self._act = _make_activation(self.activation_name)
        self._last_act = _make_activation(self.last_activation)
        self._drop = nn.Dropout(self.dropout_p) if self.use_dropout else nn.Identity()

    # ----- basic bricks -----
    def linear_layer(self, in_features: int, out_features: int,
                     *, is_last: bool = False) -> nn.Sequential:
        seq = [nn.Linear(in_features, out_features)]
        if self.use_batchnorm:
            seq.append(nn.BatchNorm1d(out_features))
        seq.append(self._last_act if is_last else self._act)
        if self.use_dropout and not is_last:
            seq.append(self._drop)
        return nn.Sequential(*seq).to(torch.float64)  # ensure double precision for all layers

    def conv_layer(self, in_channels: int, out_channels: int,
                   kernel_size: int, dilation: int,
                   *, pool_size: Optional[int] = None,
                   is_last: bool = False) -> nn.Sequential:
        if pool_size is None:
            pool_size = kernel_size
        seq = [nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)]
        if self.use_batchnorm:
            seq.append(nn.BatchNorm1d(out_channels))
        seq.append(self._last_act if is_last else self._act)
        seq.append(nn.MaxPool1d(kernel_size=pool_size))
        if self.use_dropout and not is_last:
            seq.append(self._drop)
        return nn.Sequential(*seq).to(torch.float64)  # ensure double precision for all layers

    # ----- stage builders (backwards-friendly) -----
    def fc(self, in_features: Sequence[int] = None, out_features: Sequence[int] = None, *,
           n_layers: Optional[int] = None,
           sizes: Optional[Sequence[int]] = None,
           use_last_activation: bool = False) -> nn.Sequential:
        """
        Builds a flexible FC block with the specified configuration. The user can specify either:
        1) in_features and out_features lists, which directly define the sizes of each layer, or
        2) a single sizes list, from which we will infer the in/out features for each layer.
        The number of layers is determined by the length of the sizes list or the in/out features lists, and should be consistent. The use_last_activation flag allows for optionally applying an activation function to the last layer, which can be useful for certain configurations (e.g., if the last layer is not meant to be linear).
        """
        print(f"Building FC with in_features={in_features}, out_features={out_features}, sizes={sizes}, n_layers={n_layers}")
        layers: List[nn.Module] = []
        if sizes is not None and len(sizes) > 0:
            in_f, out_f = _pair_from_sizes(sizes)
        else:
            in_f, out_f = list(in_features), list(out_features)
        if n_layers is None:
            n_layers = len(in_f)
        print(f"Building FC with in={in_f}, out={out_f}, n_layers={n_layers}, use_last_activation={use_last_activation}")
        assert n_layers == len(in_f) == len(out_f), f"FC spec length mismatch: n_layers={n_layers}, in_f={len(in_f)}, out_f={len(out_f)}"
        # Robust compatibility check: ensure each in_f[i] matches previous output shape
        for i in range(n_layers):
            if i > 0 and in_f[i] != out_f[i-1]:
                raise ValueError(f"FC layer size mismatch at layer {i}: in_features={in_f[i]} does not match previous out_features={out_f[i-1]}. Full sizes: in_f={in_f}, out_f={out_f}, sizes={sizes}")
            is_last = (i == n_layers - 1) and use_last_activation
            layers.append(self.linear_layer(in_f[i], out_f[i], is_last=is_last))

        # Add runtime assertion for input shape compatibility
        class AssertInputShape(nn.Module):
            def __init__(self, expected_in_features):
                super().__init__()
                self.expected_in_features = expected_in_features
            def forward(self, x):
                if x.shape[-1] != self.expected_in_features:
                    raise RuntimeError(f"Input tensor last dimension {x.shape[-1]} does not match expected in_features {self.expected_in_features} for FC block.")
                return x

        seq = nn.Sequential(AssertInputShape(in_f[0]), *layers)
        return seq.to(torch.float64)  # ensure double precision for all layers

    def cnn(self,
            in_channels: Sequence[int],
            out_channels: Sequence[int],
            kernel_size: Sequence[int],
            dilation: Sequence[int],
            *,
            pool_kernel_size: Optional[Sequence[Optional[int]]] = None,
            n_layers: Optional[int] = None,
            use_last_activation: bool = False) -> nn.Sequential:
        # Defensive: auto-expand single values to lists of n_layers length
        def expand(val, n):
            if isinstance(val, (list, tuple)):
                return list(val)
            return [val] * n

        # Find the minimum length among all parameter lists
        param_lengths = [
            len(in_channels) if isinstance(in_channels, (list, tuple)) else 1,
            len(out_channels) if isinstance(out_channels, (list, tuple)) else 1,
            len(kernel_size) if isinstance(kernel_size, (list, tuple)) else 1,
            len(dilation) if isinstance(dilation, (list, tuple)) else 1,
            len(pool_kernel_size) if pool_kernel_size and isinstance(pool_kernel_size, (list, tuple)) else 1
        ]
        min_len = min(param_lengths)
        # If n_layers is not set, use min_len; if set, use min(n_layers, min_len)
        n_layers = n_layers if n_layers is not None else min_len
        if n_layers > min_len:
            import warnings
            warnings.warn(f"CNN spec mismatch: Reducing n_layers from {n_layers} to {min_len} due to parameter list lengths. in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}, dilation={dilation}, pool_kernel_size={pool_kernel_size}")
            n_layers = min_len

        in_c  = expand(in_channels, n_layers)
        out_c = expand(out_channels, n_layers)
        ksz   = expand(kernel_size, n_layers)
        dil   = expand(dilation, n_layers)
        pool  = expand(pool_kernel_size if pool_kernel_size is not None else None, n_layers)

        # Validate lengths
        if not (len(in_c) == len(out_c) == len(ksz) == len(dil) == len(pool) == n_layers):
            raise ValueError(f"CNN spec length mismatch: in_channels={in_c}, out_channels={out_c}, kernel_size={ksz}, dilation={dil}, pool_kernel_size={pool}, n_layers={n_layers}")

        layers: List[nn.Module] = []
        for i in range(n_layers):
            is_last = (i == n_layers - 1) and use_last_activation
            layers.append(self.conv_layer(in_c[i], out_c[i], ksz[i], dil[i], pool_size=pool[i], is_last=is_last))
        return nn.Sequential(*layers).to(torch.float64)  # ensure double precision for all layers

    def _calculate_cnn_output_size(self, sequence_length=500):
        """
        Calculates the output size of the CNN layers given the input
        sequence length and the CNN configuration. This is necessary to determine
        the correct input size for the post-FC layers after the CNN layers.
        Works for arbitrary CNN configurations, including varying kernel sizes, 
        dilations, and pooling.

        Args:
            sequence_length (int): The input sequence length to CNN layers.

        Returns:
            int: The size of the flattened CNN output.
        """
        length = sequence_length
        for i in range(len(self.cnn_in_channels)):
            kernel_size = self.cnn_kernel_size[i] if i < len(self.cnn_kernel_size) else self.cnn_kernel_size[-1]
            dilation = self.cnn_dilation[i] if i < len(self.cnn_dilation) else self.cnn_dilation[-1]
            pool_size = self.cnn_pool_ks[i] if i < len(self.cnn_pool_ks) else self.cnn_pool_ks[-1] if self.cnn_pool_ks else kernel_size
            # Calculate the effective kernel size with dilation
            effective_kernel_size = (kernel_size - 1) * dilation + 1
            # Update length after convolution (assuming stride=1 and no padding)
            length = length - effective_kernel_size + 1
            # Update length after pooling
            length = length // pool_size
        final_out_channels = self.cnn_out_channels[-1] if self.cnn_out_channels else self.cnn_in_channels[-1]
        return final_out_channels * length
    

class BaseEncoder(BaseCoder):
    """
    Encoder for the input data to the CVAE/CAE model.
    Has flexible architecture that can include pre-FC layers, CNN layers, and post-FC layers,
    with configurable sizes and counts for each stage. The actual layers are built in 
    the forward pass
    based on the provided configuration, allowing for dynamic architectures.
    """
    def __init__(self, **kwargs):
        super(BaseEncoder, self).__init__(**kwargs)

        # -- Build layers on the fly based on config (this allows for dynamic architectures)
        # -- Do this in __init__ so that the layers are registered as part of the module, 
        # but they will be built based on the config! This allows `model.parameters()` to work correctly and include these layers, even though they are built based on the config.
        if self.has_pre_fc and (self.pre_fc_sizes or self.pre_fc_out_features):
            pre_fc_in_features = self.input_shape[0] * self.input_shape[1] if self.input_shape else self.pre_fc_in_features[0]
            pre_fc_out_features = self.pre_fc_out_features[-1] if self.pre_fc_out_features else self.pre_fc_sizes[-1]
            self.pre_fc_layers = self.fc(pre_fc_in_features, pre_fc_out_features,
                        n_layers=self.n_layers_pre_fc,
                        sizes=self.pre_fc_sizes,
                        use_last_activation=True)

        if self.has_cnn and self.cnn_in_channels and self.cnn_out_channels and self.cnn_kernel_size:
            self.cnn_layer = self.cnn(self.cnn_in_channels, self.cnn_out_channels, self.cnn_kernel_size, self.cnn_dilation,
                         pool_kernel_size=self.cnn_pool_ks,
                         n_layers=self.n_layers_cnn)

        if self.has_post_fc and (self.post_fc_sizes or self.post_fc_out_features):
            post_fc_in_features = self._calculate_cnn_output_size() if self.has_cnn else self.input_shape[0] * self.input_shape[1] if self.input_shape else self.post_fc_in_features[0]
            post_fc_out_features = self.post_fc_out_features[-1] if self.post_fc_out_features else self.post_fc_sizes[-1]
            self.post_fc_layers = self.fc(post_fc_in_features, post_fc_out_features,
                        n_layers=self.n_layers_post_fc,
                        sizes=self.post_fc_sizes,
                        use_last_activation=False)

    def forward(self, x):
        if self.has_pre_fc:
            z = x.view(x.size(0), -1)  # flatten input for FC layers
            z = self.pre_fc_layers(z)
        else:
            z = x
        # print(f"After pre-FC layers, z shape: {z.shape}")

        if self.has_cnn and self.cnn_in_channels and self.cnn_out_channels and self.cnn_kernel_size:
            if self.has_pre_fc:
                z = z.view(z.size(0), self.cnn_in_channels[0], -1)  # reshape to (B, C, L) for CNN
            else:
                z = z.view(z.size(0), self.input_shape[0], self.input_shape[1])  # reshape to (B, C, L) for CNN
            z = self.cnn_layer(z)
            z = z.view(z.size(0), -1)  # flatten CNN output for post-FC layers

        if self.has_post_fc:
            z = self.post_fc_layers(z)
        return z.view(-1, *self.input_shape)
    
class BaseDecoder(BaseCoder):
    """
    Decoder for the CVAE/CAE model. Similar flexible architecture as the encoder,
    allowing for pre-FC layers, CNN layers, and post-FC layers with configurable sizes
    and counts. The actual layers are built in the forward pass based on the provided
    configuration.
    """
    def __init__(self, **kwargs):
        super(BaseDecoder, self).__init__(**kwargs)

        if self.has_pre_fc and (self.pre_fc_sizes or self.pre_fc_in_features):
            self.pre_fc_layers = self.fc(self.pre_fc_in_features, self.pre_fc_out_features,
                        n_layers=self.n_layers_pre_fc,
                        sizes=self.pre_fc_sizes)

        if self.has_cnn and self.cnn_in_channels:
            self.cnn_layers = self.cnn(self.cnn_in_channels, self.cnn_out_channels,
                         self.cnn_kernel_size, self.cnn_dilation,
                         pool_kernel_size=self.cnn_pool_ks,
                         n_layers=self.n_layers_cnn)

        if self.has_post_fc and (self.post_fc_sizes or self.post_fc_out_features):
            post_fc_in_features = self._calculate_cnn_output_size() if self.has_cnn else self.input_shape[0] * self.input_shape[1] if self.input_shape else self.post_fc_in_features[0]
            self.post_fc_layers = self.fc(post_fc_in_features, self.post_fc_out_features,
                        n_layers=self.n_layers_post_fc,
                        sizes=self.post_fc_sizes,
                        use_last_activation=False)

    def forward(self, z: torch.Tensor, y: torch.Tensor, y_embed: Optional[torch.Tensor] = None):
        assert z.size(1) == self.latent_dim, "z latent_dim mismatch"
        assert y.size(1) == self.num_classes, "y num_classes mismatch"
        y_cat = y_embed if y_embed is not None else y
        z = torch.cat([z, y_cat], dim=1)

        if self.has_pre_fc:
            z = self.pre_fc_layers(z)

        if self.has_cnn and self.cnn_in_channels:
            z = z.view(z.size(0), self.cnn_in_channels[0], -1)
            z = self.cnn_layers(z)
            z = z.view(z.size(0), -1)

        if self.has_post_fc and (self.post_fc_sizes or self.post_fc_out_features):
            z = self.post_fc_layers(z)
        return z.view(-1, *self.input_shape)
    
class BaseConditional(BaseCoder):
    """
    Base class for conditional encoder for the input parameter labels.
    Only contains FC layers and no CNN layers, since the labels are typically low-dimensional vectors.
    The architecture is flexible and can include multiple FC layers with configurable sizes and counts.
    The actual layers are built in the forward pass based on the provided configuration, allowing for dynamic architectures.
    """
    def __init__(self, **kwargs):
        super(BaseConditional, self).__init__(**kwargs)
        self.has_pre_fc=True
        self.has_cnn=False
        self.has_post_fc=False

        pre_fc_in_features = self.input_shape[0] if self.input_shape else self.pre_fc_in_features[0]
        pre_fc_out_features = self.pre_fc_out_features[-1] if self.pre_fc_out_features else self.pre_fc_sizes[-1]
       
        self.pre_fc_layers = self.fc(pre_fc_in_features, pre_fc_out_features,
                    n_layers=self.n_layers_pre_fc,
                    sizes=self.pre_fc_sizes,
                    use_last_activation=False)

    def forward(self, y):
        z = y.view(y.size(0), -1)  # flatten input for FC layers
        z = self.pre_fc_layers(z)
        return z.view(-1, self.latent_dim)  # ensure output shape is (B, latent_dim)
    


class TwoC2E1D(nn.Module):
    """
    Conditional Variational Autoencoder (CVAE) implementation based on my 
    paper. We basically keep everything the same, e.g. loss function and
    what goes into the decoder, and only use the variational model.
    What is flexible is the number of layers in encoders / conditionals and
    the size of each of those layers. Plus, the activation function is
    also optimized during hyper-parameter tuning.
    Aim is to get the best model performance for the configuration I have
    in CVAE-Paper-I !

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
    def __init__(self, input_shape=(2,8191), num_classes=4, key_shape=(2,2), \
                 labels_mean=None, labels_std=None, paramsnorm=False, \
                 latent_dim_x=8, latent_dim_key=3, MODEL_CONFIG=None, **kwargs):
        super(TwoC2E1D, self).__init__()

        # Override hyperparameters with MODEL_CONFIG values if provided
        # This allows for flexible model configuration while maintaining default values.
        if MODEL_CONFIG is not None:
            self.MODEL_CONFIG = MODEL_CONFIG
            input_shape = MODEL_CONFIG.get('input_shape', input_shape)
            num_classes = MODEL_CONFIG.get('num_classes', num_classes)
            key_shape = MODEL_CONFIG.get('key_shape', key_shape)
            latent_dim_x = MODEL_CONFIG.get('latent_dim_x', latent_dim_x)
            latent_dim_key = MODEL_CONFIG.get('latent_dim_key', latent_dim_key)
            labels_mean = MODEL_CONFIG.get('labels_mean', labels_mean)
            labels_std = MODEL_CONFIG.get('labels_std', labels_std)
            paramsnorm = MODEL_CONFIG.get('paramsnorm', paramsnorm)
            self.activation_name = MODEL_CONFIG.get('activation', 'relu')
            self.beta = MODEL_CONFIG.get('beta', 0.1)  # default beta value for KL divergence loss
            self.decoder_input_type = MODEL_CONFIG.get('decoder_input_type', 'concat')
            self.embed_labels_in_decoder = MODEL_CONFIG.get('embed_labels_in_decoder', False)
            logging.info(f"MODEL_CONFIG provided. Using hyperparameters from MODEL_CONFIG: {MODEL_CONFIG}")
        else:
            logging.info("No MODEL_CONFIG provided. Using default hyperparameter values.")

        # If MODEL_CONFIG is provided, it should contain all necessary hyperparameters.
        # If not provided, the default values will be used.
        self.latent_dim_x = latent_dim_x * 2       # latent mean and logvar
        self.latent_dim_key = latent_dim_key * 2   # latent mean and logvar
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.key_shape = key_shape

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
            logging.warning("labels_mean and labels_std are provided but paramsnorm is False. \
                These will be ignored since input param normalization is NOT enabled.")
        else:
            logging.info("Input parameter normalization is NOT enabled. \
                The model will use the raw labels without normalization.")

        # The actual layers will be built in forward() based on the config
        self.encoder_x = BaseEncoder(latent_dim_x=self.latent_dim_x,
                                       input_shape=self.input_shape,
                                       num_classes=self.num_classes,
                                       n_layers=1,
                                       n_layers_cnn=2,
                                       has_pre_fc=False,
                                       has_cnn=True,
                                       has_post_fc=True,
                                       activation_name=self.activation_name,
                                       cnn_in_channels=[2, 16],
                                       cnn_out_channels=[16, 32],
                                       cnn_kernel_size=[5, 5],
                                       cnn_dilation=[1, 1],
                                       cnn_pool_kernel_size=[4, 4],
                                       post_fc_sizes=[512, self.latent_dim_x],)
        self.encoder_key = BaseEncoder(latent_dim_x=self.latent_dim_key,
                                          input_shape=self.key_shape,
                                          num_classes=self.num_classes,
                                          n_layers = 3,
                                          has_cnn=False,
                                          has_post_fc=False,
                                          activation_name=self.activation_name,
                                          pre_fc_sizes=[self.key_shape[0] * self.key_shape[1], 64, 64, self.latent_dim_key],
                                          )
        self.decoder = BaseDecoder(latent_dim=self.latent_dim_x + self.latent_dim_key,  # e.g. z1 + z1prime
                                   input_shape=self.input_shape,
                                   num_classes=self.num_classes,
                                   n_layers=1,
                                   n_layers_cnn=3,                                   
                                   has_pre_fc=True,
                                   has_cnn=True,
                                   has_post_fc=True,
                                   activation_name=self.activation_name,
                                   pre_fc_sizes=[self.latent_dim_x + self.latent_dim_key + self.num_classes, 512],
                                   cnn_in_channels=[64, 32, 16],
                                   cnn_out_channels=[32, 16, self.input_shape[0]],
                                   cnn_kernel_size=[5, 5, 5],
                                   cnn_dilation=[1, 1, 1],
                                   cnn_pool_kernel_size=[4, 4, 4],
                                   post_fc_sizes=[512, self.input_shape[0] * self.input_shape[1]],
                                   )
        self.conditional_x = BaseConditional(latent_dim=self.latent_dim_x,  # z1 mean and logvar
                                            input_shape=(self.num_classes,),
                                            num_classes=self.num_classes,
                                            n_layers=4, 
                                            activation_name=self.activation_name,
                                            pre_fc_sizes=[self.num_classes, 128, 128, 128, self.latent_dim_x])
        self.conditional_key = BaseConditional(latent_dim=self.latent_dim_key,  # z1prime mean and logvar
                                              input_shape=(self.num_classes,),
                                              num_classes=self.num_classes,
                                              n_layers=4,
                                              activation_name=self.activation_name,
                                              pre_fc_sizes=[self.num_classes, 128, 128, 128, self.latent_dim_key])
        
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
        h = self.encoder_x(x)
        z_mean, z_log_var = torch.chunk(h, 2, dim=1)
        return z_mean, z_log_var
    
    def encode_key(self, keys, labels):
        h = self.encoder_key(keys)
        z_mean, z_log_var = torch.chunk(h, 2, dim=1)
        return z_mean, z_log_var
    
    def encode_label_for_x(self, labels):
        h = self.conditional_x(labels)
        z_mean, z_log_var = torch.chunk(h, 2, dim=1)
        return z_mean, z_log_var
    
    def encode_label_for_key(self, labels):
        h = self.conditional_key(labels)
        z_mean, z_log_var = torch.chunk(h, 2, dim=1)
        return z_mean, z_log_var
    
    def decode(self, z, y_embed):
        recon_x = self.decoder(z, y_embed)
        return recon_x.view(-1, *self.input_shape)

    def reparameterize(self, z_mean, z_log_var):
        # This is the variational part of the VAE
        std = torch.exp(0.5 * z_log_var)
        # -- TODO: set random seed for reproducibility
        # torch.manual_seed(42)
        eps = torch.randn_like(std)
        return z_mean + eps * std
    
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

    def forward(self, x: torch.Tensor, y: torch.Tensor, keys: torch.Tensor):
        """
        The model learns to have all latent representations almost the same.
        We only pass in the `x` and `keys` latent representations to the decoder, 
        and not the label-conditioned latents, because the labels are already 
        passed in as input to the decoder and thus the model can learn to encode 
        all necessary information about the labels in the decoder input itself, 
        without needing separate label-conditioned latents.
        Further, the input data latents are trained to be equal to the label-conditioned latents 
        for the data, and similarly for the keys, through the latent loss term in the loss function, 
        which encourages all latent representations to be similar and thus forces the model 
        to learn a consistent latent space that captures both the data and label information.
        Later during inference, we will only pass the label-conditioned latents to the decoder, 
        since we won't have access to the input data or keys, and the model will be able to 
        generate outputs based on the learned latent space and the provided labels. Since,
        during training, the input data latents are trained to be similar to the label-conditioned latents,
        the model will learn to generate outputs that are consistent with the labels even when 
        only the label conditioned latents are provided during inference, thus enabling the model 
        to generalize well to unseen labels and generate meaningful outputs based on the learned 
        relationships between the data and the labels.
        """
        zx_mu, zx_logvar = self.encode_x(x, y)
        zy_mu, zy_logvar = self.encode_label_for_x(y)
        zkey_mu, zkey_logvar = self.encode_key(keys, y)
        zykey_mu, zykey_logvar = self.encode_label_for_key(y)

        z_x = self.reparameterize(zx_mu, zx_logvar)
        z_key = self.reparameterize(zkey_mu, zkey_logvar)

        # TODO: Can input embeddings for the labels!
        if self.embed_labels_in_decoder:
            y_embed = self.conditional_x(y)  # Use the label-conditioned encoder for x as the label embedding
        else:
            y_embed = y  # Use raw labels as input to the decoder

        # Select decoder input based on the specified type
        if self.decoder_input_type == 'sum':
            z = z_x + z_key
        elif self.decoder_input_type == 'concat':
            z = torch.cat([z_x, z_key], dim=1)
        elif self.decoder_input_type == 'onlyzx':
            z = z_x
        elif self.decoder_input_type == 'onlyzkey':
            z = z_key
        elif self.decoder_input_type == 'weighted_sum':
            alpha = 0.5  # This can be a hyperparameter to tune
            z = alpha * z_x + (1 - alpha) * z_key
        elif self.decoder_input_type == 'concat_all':
            z = torch.cat([z_x, z_key, zy_mu, zykey_mu], dim=1)
        else:
            z = torch.cat([z_x, z_key], dim=1)  # default to concat if unknown type
            logging.warning(f"Unknown decoder_input_type '{self.decoder_input_type}'. Defaulting to concatenation of z_x and z_key.")
        
        recon_x = self.decode(z, y_embed)
        zvars = [zx_mu, zx_logvar, zy_mu, zy_logvar, zkey_mu, zkey_logvar, zykey_mu, zykey_logvar]
        return (recon_x, zvars)
    

    def loss_function(self, x, x_recon, zvars) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        zx_mu, zx_logvar, zy_mu, zy_logvar, \
            zkey_mu, zkey_logvar, zykey_mu, zykey_logvar = zvars
        logging.debug(f'zx_mu={zx_mu}, zx_logvar={zx_logvar}, zy_mu={zy_mu}, zy_logvar={zy_logvar}, \
            zkey_mu={zkey_mu}, zkey_logvar={zkey_logvar}, zykey_mu={zykey_mu}, zykey_logvar={zykey_logvar}')

        # Reconstruction loss (e.g., Binary Cross-Entropy or MSE)
        # TODO: What is the `reduction` thing doing here?
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        logging.info(f"Reconstruction Loss: {recon_loss.item()}")

        # KL divergence for each latent space
        kl_loss_zx = self.latent_loss(zx_mu, zx_logvar)
        kl_loss_zy = self.latent_loss(zy_mu, zy_logvar)
        kl_loss_zkey = self.latent_loss(zkey_mu, zkey_logvar)
        kl_loss_zykey = self.latent_loss(zykey_mu, zykey_logvar)
        # Print KL divergence losses for debugging
        logging.info(f"KL Loss zx: {kl_loss_zx.item()}, KL Loss zy: {kl_loss_zy.item()}, "
            f"KL Loss zkey: {kl_loss_zkey.item()}, KL Loss zykey: {kl_loss_zykey.item()}")

        # Latent loss between encoders
        ll1 = self.latent_loss_between_encoders(zx_mu, zx_logvar, zy_mu, zy_logvar)
        ll2 = self.latent_loss_between_encoders(zkey_mu, zkey_logvar, zykey_mu, zykey_logvar)

        # Total loss
        kl_loss = kl_loss_zx + kl_loss_zy + kl_loss_zkey + kl_loss_zykey + ll1 + ll2
        total_loss = recon_loss + self.beta * kl_loss
        total_loss = torch.tensor(total_loss, dtype=torch.float64)  # ensure total loss is in double precision
        return (total_loss, recon_loss, kl_loss)
    
    def _save_model_config(self, filepath=None, **kwargs):
        """
        Saves the model configuration to JSON file.
        """
        if filepath is None:
            filepath = 'model_config.json'
        if not '.json' in filepath:
            logging.warning(f"Model configuration file should be a JSON file. Adding '.json' extension to {filepath}.")
            filepath += '.json'
        if not hasattr(self, 'MODEL_CONFIG'):
            logging.warning("MODEL_CONFIG attribute not found. Creating a new MODEL_CONFIG dictionary to save hyperparameters.")
            self.MODEL_CONFIG = {}
        # Add additional hyperparameters to MODEL_CONFIG before saving
        self.MODEL_CONFIG.update({
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'key_shape': self.key_shape,
            'latent_dim_x': self.latent_dim_x // 2,  # divide by 2 to get original latent dim before doubling for mean and logvar
            'latent_dim_key': self.latent_dim_key // 2,  # divide by 2 to get original latent dim before doubling for mean and logvar
            'labels_mean': getattr(self, 'labels_mean', None),
            'labels_std': getattr(self, 'labels_std', None),
            'paramsnorm': hasattr(self, 'labels_mean') and hasattr(self, 'labels_std'),
            'activation': self.activation_name,
            'beta': self.beta,
            'decoder_input_type': self.decoder_input_type,
            'embed_labels_in_decoder': self.embed_labels_in_decoder,
            'n_layers': {
                'pre_fc': self.n_layers_pre_fc,
                'cnn': self.n_layers_cnn,
                'post_fc': self.n_layers_post_fc
            },
            'pre_fc_sizes': self.pre_fc_sizes,
            'cnn_in_channels': self.cnn_in_channels,
            'cnn_out_channels': self.cnn_out_channels,
            'cnn_kernel_size': self.cnn_kernel_size,
            'cnn_dilation': self.cnn_dilation,
            'cnn_pool_ks': self.cnn_pool_ks,
        })
        # Update config with model architecture and parameters details
        self.MODEL_CONFIG.update({
            'model_architecture': str(self),
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        })
        # Save other supplied kwargs to MODEL_CONFIG
        self.MODEL_CONFIG.update(kwargs)
        torch.save(self.MODEL_CONFIG, filepath)
        logging.info(f"Model configuration saved to {filepath}")
        
