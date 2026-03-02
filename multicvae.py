# Notes:
# - Keeps the original structure (BaseCoder → Enc/Dec → CVAE) but streamlines
#   and generalizes with optional Conditioners (C) and tiny utilities.
# - Backwards-compatible kwargs are preserved where possible (e.g., pre_fc_in_features,
#   pre_fc_out_features, etc.).
# - We can:
#     * Select activations (relu/gelu/silu/elu/leaky_relu) per coder
#     * Toggle/use dropout & batchnorm
#     * Configure CNN/FC layer counts and sizes per stage (pre_fc / cnn / post_fc)
#     * Add Conditioners to embed labels y before concatenation
#     * Build models by code like "2C2E1D" (= 2 Conditioners, 2 Encoders, 1 Decoder)

from init import *  # expects torch, nn, etc. to be available
from utils import *
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
    """Given sizes like [a, b, c], return in=[a, b], out=[b, c]."""
    sizes = list(sizes)
    assert len(sizes) >= 2, "sizes must have len >= 2"
    in_f = sizes[:-1]
    out_f = sizes[1:]
    return in_f, out_f


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


# -----------------
# Core Coder Blocks
# -----------------
class BaseCoder(nn.Module):
    """General base coder inherited by encoders/decoders.

    Minimal-change spirit: keeps original API keys while adding nicer knobs.
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

        # Layer counts (defaults kept)
        self.n_layers            = kwargs.get('n_layers', None)
        self.n_layers_cnn        = kwargs.get('n_layers_cnn', self.n_layers)
        self.n_layers_fc         = kwargs.get('n_layers_fc', self.n_layers)
        self.n_layers_pre_fc     = kwargs.get('n_layers_pre_fc', self.n_layers_fc)
        self.n_layers_post_fc    = kwargs.get('n_layers_post_fc', self.n_layers_fc)

        # FC sizing (legacy-compatible)
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

        # Keep small, Jennie-simple layers ready
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
        return nn.Sequential(*seq)

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
        return nn.Sequential(*seq)

    # ----- stage builders (backwards-friendly) -----
    def fc(self, in_features: Sequence[int], out_features: Sequence[int], *,
           n_layers: Optional[int] = None,
           sizes: Optional[Sequence[int]] = None,
           use_last_activation: bool = False) -> nn.Sequential:
        """Builds an MLP. You may specify either (in_features, out_features)
        or a single `sizes=[in, h1, ..., out]`. The original API is preserved.
        """
        layers: List[nn.Module] = []
        if sizes is not None and len(sizes) > 0:
            in_f, out_f = _pair_from_sizes(sizes)
        else:
            in_f, out_f = list(in_features), list(out_features)
        if n_layers is None:
            n_layers = len(in_f)
        assert n_layers == len(in_f) == len(out_f), "FC spec length mismatch"
        for i in range(n_layers):
            is_last = (i == n_layers - 1) and use_last_activation
            layers.append(self.linear_layer(in_f[i], out_f[i], is_last=is_last))
        return nn.Sequential(*layers)

    def cnn(self,
            in_channels: Sequence[int],
            out_channels: Sequence[int],
            kernel_size: Sequence[int],
            dilation: Sequence[int],
            *,
            pool_kernel_size: Optional[Sequence[Optional[int]]] = None,
            n_layers: Optional[int] = None,
            use_last_activation: bool = False) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_c  = list(in_channels)
        out_c = list(out_channels)
        ksz   = list(kernel_size)
        dil   = list(dilation)
        if pool_kernel_size is None or len(pool_kernel_size) == 0:
            pool = [None] * len(in_c)
        else:
            pool = list(pool_kernel_size)
        assert len(in_c) == len(out_c) == len(ksz) == len(dil) == len(pool), "CNN spec length mismatch"
        if n_layers is None:
            n_layers = len(in_c)
        for i in range(n_layers):
            is_last = (i == n_layers - 1) and use_last_activation
            layers.append(self.conv_layer(in_c[i], out_c[i], ksz[i], dil[i], pool_size=pool[i], is_last=is_last))
        return nn.Sequential(*layers)


# -----------------
# Encoders/Decoders
# -----------------
class BaseEncoder(BaseCoder):
    def __init__(self, **kwargs):
        super(BaseEncoder, self).__init__(**kwargs)
        self.concat_xy_before = kwargs.get('concat_xy_before', False)

    def forward(self, x: torch.Tensor, y: torch.Tensor, y_embed: Optional[torch.Tensor] = None):
        y_cat = y_embed if y_embed is not None else y
        if self.concat_xy_before:
            x = torch.cat([x.view(x.size(0), -1), y_cat], dim=1)
        else:
            x = x.view(x.size(0), -1)

        if self.has_pre_fc and (self.pre_fc_sizes or self.pre_fc_in_features):
            x = self.fc(self.pre_fc_in_features, self.pre_fc_out_features,
                        n_layers=self.n_layers_pre_fc,
                        sizes=self.pre_fc_sizes)(x)

        if self.has_cnn and self.cnn_in_channels:
            x = x.view(x.size(0), self.cnn_in_channels[0], -1)
            x = self.cnn(self.cnn_in_channels, self.cnn_out_channels,
                         self.cnn_kernel_size, self.cnn_dilation,
                         pool_kernel_size=self.cnn_pool_ks,
                         n_layers=self.n_layers_cnn)(x)
            x = x.view(x.size(0), -1)

        if not self.concat_xy_before:
            x = torch.cat([x.view(x.size(0), -1), y_cat], dim=1)

        if self.has_post_fc and (self.post_fc_sizes or self.post_fc_in_features):
            x = self.fc(self.post_fc_in_features, self.post_fc_out_features,
                        n_layers=self.n_layers_post_fc,
                        sizes=self.post_fc_sizes,
                        use_last_activation=False)(x)

        return x.view(-1, self.latent_dim)


class BaseDecoder(BaseCoder):
    def forward(self, z: torch.Tensor, y: torch.Tensor, y_embed: Optional[torch.Tensor] = None):
        assert z.size(1) == self.latent_dim, "z latent_dim mismatch"
        assert y.size(1) == self.num_classes, "y num_classes mismatch"
        y_cat = y_embed if y_embed is not None else y
        z = torch.cat([z, y_cat], dim=1)

        if self.has_pre_fc and (self.pre_fc_sizes or self.pre_fc_in_features):
            z = self.fc(self.pre_fc_in_features, self.pre_fc_out_features,
                        n_layers=self.n_layers_pre_fc,
                        sizes=self.pre_fc_sizes)(z)

        if self.has_cnn and self.cnn_in_channels:
            z = z.view(z.size(0), self.cnn_in_channels[0], -1)
            z = self.cnn(self.cnn_in_channels, self.cnn_out_channels,
                         self.cnn_kernel_size, self.cnn_dilation,
                         pool_kernel_size=self.cnn_pool_ks,
                         n_layers=self.n_layers_cnn)(z)
            z = z.view(z.size(0), -1)

        if self.has_post_fc and (self.post_fc_sizes or self.post_fc_in_features):
            z = self.fc(self.post_fc_in_features, self.post_fc_out_features,
                        n_layers=self.n_layers_post_fc,
                        sizes=self.post_fc_sizes,
                        use_last_activation=False)(z)

        return z.view(-1, *self.input_shape)


# ----------------
# Conditioner (C)
# ----------------
class Conditioner(nn.Module):
    """
    Encodes the label y into a continuous embedding in the latent space.

    Args:
        in_dim: y input dim (usually num_classes)
        out_dim: embedding dim
        sizes: optional sizes=[in, ..., out] to explicitly configure layers
        activation: activation for hidden layers
        use_batchnorm, dropout_p: regularization knobs
    """
    def __init__(self, in_dim: int, out_dim: int, *,
                 sizes: Optional[Sequence[int]] = None,
                 activation: str = 'relu',
                 use_batchnorm: bool = False,
                 dropout_p: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act = _make_activation(activation)
        drop = nn.Dropout(dropout_p) if (dropout_p and dropout_p > 0) else nn.Identity()

        if sizes is None or len(sizes) == 0:
            sizes = [in_dim, out_dim]
        in_f, out_f = _pair_from_sizes(sizes)
        layers: List[nn.Module] = []
        for i, (a, b) in enumerate(zip(in_f, out_f)):
            layers.append(nn.Linear(a, b))
            if i < len(in_f) - 1:  # no act/bn/drop on the last by default
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(b))
                layers.append(act)
                layers.append(drop)
        self.net = nn.Sequential(*layers)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)


# ------------------------------------------
# Higher-order models: flexible enc/C/dec
# ------------------------------------------
class BaseCVAE(nn.Module):
    """Base class for (C)VAEs with variable # of Encoders / Conditioners / Decoders.
    Backwards-compatible: you can ignore conditioners by leaving n_conditioners=0.
    """
    def __init__(self, **kwargs):
        super(BaseCVAE, self).__init__()

        # Counts
        self.n_conditioners = kwargs.get('n_conditioners', 0)
        self.n_encoders     = kwargs.get('n_encoders', 2)
        self.n_decoders     = kwargs.get('n_decoders', 2)

        # Per-stage toggles / layer-count defaults (kept for convenience)
        self.encoder_has_cnn = kwargs.get('encoder_has_cnn', True)
        self.decoder_has_cnn = kwargs.get('decoder_has_cnn', True)
        self.encoder_has_fc  = kwargs.get('encoder_has_fc', True)
        self.decoder_has_fc  = kwargs.get('decoder_has_fc', True)

        self.n_layers            = kwargs.get('n_layers', 2)
        self.n_layers_cnn        = kwargs.get('n_layers_cnn', self.n_layers)
        self.n_layers_fc         = kwargs.get('n_layers_fc', self.n_layers)
        self.n_layers_cnn_encoder = kwargs.get('n_layers_cnn_encoder', self.n_layers_cnn)
        self.n_layers_cnn_decoder = kwargs.get('n_layers_cnn_decoder', self.n_layers_cnn)
        self.n_layers_fc_encoder  = kwargs.get('n_layers_fc_encoder', self.n_layers_fc)
        self.n_layers_fc_decoder  = kwargs.get('n_layers_fc_decoder', self.n_layers_fc)

        # Optional shared conditioner config
        cond_dim = kwargs.get('cond_dim', None)
        conditioner_sizes = kwargs.get('conditioner_sizes', None)
        conditioner_activation = kwargs.get('conditioner_activation', 'relu')
        conditioner_use_bn = kwargs.get('conditioner_use_batchnorm', False)
        conditioner_dropout = kwargs.get('conditioner_dropout_p', 0.0)
        num_classes = kwargs.get('num_classes')

        # Build conditioners
        self.conditioners = nn.ModuleList()
        if self.n_conditioners > 0:
            assert cond_dim is not None, "Provide cond_dim when using Conditioners"
            for _ in range(self.n_conditioners):
                self.conditioners.append(
                    Conditioner(
                        in_dim=num_classes,
                        out_dim=cond_dim,
                        sizes=conditioner_sizes,
                        activation=conditioner_activation,
                        use_batchnorm=conditioner_use_bn,
                        dropout_p=conditioner_dropout,
                    )
                )

        # Build encoders/decoders
        enc_kwargs = dict(**kwargs)
        dec_kwargs = dict(**kwargs)
        
        # Allow per-encoder latent dimensions
        encoder_latent_dims = kwargs.get('encoder_latent_dims', None)
        if encoder_latent_dims is None:
            encoder_latent_dims = [kwargs.get('latent_dim')] * self.n_encoders
        
        enc_kwargs.update(dict(
            has_cnn=self.encoder_has_cnn,
            has_pre_fc=self.encoder_has_fc,
            has_post_fc=self.encoder_has_fc,
            n_layers=self.n_layers,
            n_layers_cnn=self.n_layers_cnn_encoder,
            n_layers_fc=self.n_layers_fc_encoder,
        ))
        
        # Store for later use
        self.encoder_latent_dims = encoder_latent_dims
        dec_kwargs.update(dict(
            has_cnn=self.decoder_has_cnn,
            has_pre_fc=self.decoder_has_fc,
            has_post_fc=self.decoder_has_fc,
            n_layers=self.n_layers,
            n_layers_cnn=self.n_layers_cnn_decoder,
            n_layers_fc=self.n_layers_fc_decoder,
        ))

        self.encoders = nn.ModuleList([BaseEncoder(**enc_kwargs) for _ in range(self.n_encoders)])
        self.decoders = nn.ModuleList([BaseDecoder(**dec_kwargs) for _ in range(self.n_decoders)])

    # ---- helpers to access C/E/D ----
    def condition(self, y: torch.Tensor, idx: int = 0) -> Optional[torch.Tensor]:
        """
        This acts as a separate encoder for labels y, which are contatenated with
        the inputs, but are also separately transformed via Conditioners. So that,
        later, the inputs can be removed and only the conditioned labels can be used
        to guide the generation.
        """
        if self.n_conditioners == 0:
            return None
        return self.conditioners[idx % self.n_conditioners](y)

    def encode(self, x: torch.Tensor, y: torch.Tensor, encoder_idx: int = 0) -> torch.Tensor:
        return self.encoders[encoder_idx](x, y)

    def decode(self, z: torch.Tensor, y: torch.Tensor, decoder_idx: int = 0) -> torch.Tensor:
        return self.decoders[decoder_idx](z, y)

    def forward(self, x: torch.Tensor, y: torch.Tensor,
                encoder_idx: int = 0, decoder_idx: int = 0):
        z = self.encode(x, y, encoder_idx)
        out = self.decode(z, y, decoder_idx)
        return out


# --------------------------------------
# Intermediate Enc/Dec presets (optional)
# --------------------------------------
class NLayerEncoder(BaseEncoder):
    """Encoder with explicit (n_pre_fc, n_cnn, n_post_fc) and sizes.

    Example:
        NLayerEncoder(
            n_layers_pre_fc=2, pre_fc_sizes=[in_dim, 128, 64],
            n_layers_cnn=2,
            cnn_in_channels=[1, 16], cnn_out_channels=[16, 32],
            cnn_kernel_size=[5, 3], cnn_dilation=[1, 1],
            n_layers_post_fc=1, post_fc_sizes=[post_in, latent_dim],
            activation='silu', use_batchnorm=True, dropout_p=0.1,
            ...
        )
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class NLayerDecoder(BaseDecoder):
    """Decoder with explicit (n_pre_fc, n_cnn, n_post_fc) and sizes.
    Mirrors NLayerEncoder configuration options.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# ----------------------------------------------------
# Friendly factory: build from code like "2C2E1D"
# ----------------------------------------------------
class FlexibleCVAE(BaseCVAE):
    pass  # BaseCVAE already implements the flexible wiring


def build_model_from_code(code: str, **kwargs) -> BaseCVAE:
    """Parse codes like "2C2E1D" → 2 Conditioners, 2 Encoders, 1 Decoder.
    Fallbacks: missing sections default to 0C / 1E / 1D where sensible.
    """
    code = code.upper().strip()
    import re
    nC = re.search(r"(\d+)C", code)
    nE = re.search(r"(\d+)E", code)
    nD = re.search(r"(\d+)D", code)
    n_conditioners = int(nC.group(1)) if nC else kwargs.get('n_conditioners', 0)
    n_encoders     = int(nE.group(1)) if nE else kwargs.get('n_encoders', 1)
    n_decoders     = int(nD.group(1)) if nD else kwargs.get('n_decoders', 1)

    return FlexibleCVAE(
        n_conditioners=n_conditioners,
        n_encoders=n_encoders,
        n_decoders=n_decoders,
        **kwargs,
    )


class BaseTwoC2E1D(BaseCVAE):
    """Concrete: 2 Conditioners, 2 Encoders, 1 Decoder (keeps original spirit).
    If you also want 2 Conditioners, call build_model_from_code("2C2E1D", ...).
    """
    def __init__(self, **kwargs):
        super(BaseTwoC2E1D, self).__init__(n_conditioners=2, n_encoders=2, n_decoders=1, **kwargs)

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
        # print('__call__')
        return self.forward(x, labels, keys)

    def forward(self, x, labels, keys=None):
        """
        Since there are two encoders and two conditioners, we encode the input x and keys separately,
        and also condition the labels separately for each encoder. Then we concatenate all the latent
        representations and pass them to the decoder to reconstruct the input. 
        The method returns the reconstructed output and the latent variables
        for both encoders and conditioners, which can be used for computing the loss.

        Arguments:
            x (Tensor): The input data to be encoded and reconstructed.
            labels (Tensor): The conditional labels that guide the encoding and decoding process.
            keys (Tensor, optional): Additional input data that can be encoded separately. 
                Defaults to None.

        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]:
                - The first element is the reconstructed output from the decoder.
                - The second element is a tuple containing mean and log variance of latent variables
                  for both encoders and conditioners: 
                    (z1_mean, z1_logvar, z1p_mean, z1p_logvar, z2_mean, z2_logvar, z2p_mean, z2p_logvar).
        """
        labels = self.normalize_labels(labels)
        z1 = self.encode(x, labels, encoder_idx=0, conditioner_idx=0)
        z1_mean, z1_logvar = z1.chunk(2, dim=1)
        z1p = self.condition(labels, idx=0)
        z1p_mean, z1p_logvar = z1p.chunk(2, dim=1)
        z2 = self.encode(keys, labels, encoder_idx=1, conditioner_idx=1)
        z2_mean, z2_logvar = z2.chunk(2, dim=1)
        z2p = self.condition(labels, idx=1)
        z2p_mean, z2p_logvar = z2p.chunk(2, dim=1)
        z = torch.cat([z1, z1p, z2, z2p], dim=1)
        x_recon = self.decode(z, labels, decoder_idx=0)
        zvars = (z1_mean, z1_logvar, z1p_mean, z1p_logvar, z2_mean, z2_logvar, z2p_mean, z2p_logvar)
        return (x_recon, zvars)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from N(mu, var) from N(0,1)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def latent_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """ 
        Compute the KL divergence loss between the learned latent distribution
        and the standard normal distribution.
        """
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        kll = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return torch.mean(kll)
    
    def latent_loss_bet_encoders(self, z1_vars, z1p_vars, z2_vars, z2p_vars):
        """ 
        Compute the KL divergence loss between two encoder distributions.
        """
        mu1, logvar1 = z1_vars
        mu2, logvar2 = z2_vars
        logvar1 = torch.clamp(logvar1, min=-10.0, max=10.0)
        logvar2 = torch.clamp(logvar2, min=-10.0, max=10.0)
        kll = 0.5 * torch.sum(
            logvar2 - logvar1 + 
            (torch.exp(logvar1) + (mu1 - mu2).pow(2)) / torch.exp(logvar2) - 1,
            dim=1
        )
        return torch.mean(kll)
    
    def loss_function(self, x_target, x_recon, zvars, beta=0.1):
        """ 
        Loss function combining reconstruction loss and KL divergence.
        """
        recon_loss = F.mse_loss(x_recon, x_target, reduction='mean')
        z1_mean, z1_logvar, z1p_mean, z1p_logvar, z2_mean, z2_logvar, z2p_mean, z2p_logvar = zvars
        # latent loss between latent and standard normal distribution
        kl_loss_enc1 = self.latent_loss(z1_mean, z1_logvar)
        kl_loss_cond1 = self.latent_loss(z1p_mean, z1p_logvar)
        kl_loss_enc2 = self.latent_loss(z2_mean, z2_logvar)
        kl_loss_cond2 = self.latent_loss(z2p_mean, z2p_logvar)
        # latent loss between encoders
        kll1 = self.latent_loss_bet_encoders(z1_mean, z1_logvar, z2_mean, z2_logvar)
        # latent loss between conditioners
        kll2 = self.latent_loss_bet_encoders(z1p_mean, z1p_logvar, z2p_mean, z2p_logvar)
        kl_loss = kl_loss_enc1 + kl_loss_cond1 + kl_loss_enc2 + kl_loss_cond2 + kll1 + kll2
        total_loss = recon_loss + beta * kl_loss
        return (total_loss, recon_loss, kl_loss)
    

class TwoC2E1D(BaseTwoC2E1D):
    """
    Customized 2C2E1D model with my own preprocessing
    """
    def __init__(self, labels_mean, labels_std, **kwargs):
        super(TwoC2E1D, self).__init__(**kwargs)

        self.register_buffer('labels_mean', torch.tensor(labels_mean))
        self.register_buffer('labels_std', torch.tensor(labels_std))

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
        return (labels - self.labels_mean) / self.labels_std
    


# -----------------------------
# Tiny usage sketch (reference)
# -----------------------------
"""
Example 1 — classic 2E1D with custom layer counts & features

model = TwoC2E1D(
    # shared core dims
    input_shape=(1, 2048),
    latent_dim=128,
    num_classes=10,

    # encoder/decoder stage toggles
    encoder_has_cnn=True,
    decoder_has_cnn=True,
    encoder_has_fc=True,
    decoder_has_fc=True,

    # layer counts (per stage)
    n_layers_cnn_encoder=2,
    n_layers_cnn_decoder=2,
    n_layers_fc_encoder=2,   # used for pre+post unless you override below
    n_layers_fc_decoder=2,
    n_layers_pre_fc=2,
    n_layers_post_fc=1,

    # activations & regularization
    activation='silu',
    use_batchnorm=True,
    dropout_p=0.1,

    # FC sizes (either pair lists or single sizes=[...])
    pre_fc_sizes=[(1*2048)+10, 512, 256],
    post_fc_sizes=[256+10, 128],  # final must reach latent_dim on encoder, or input dims on decoder

    # CNN specs
    cnn_in_channels=[1, 16],
    cnn_out_channels=[16, 32],
    cnn_kernel_size=[7, 5],
    cnn_dilation=[1, 1],
    cnn_pool_kernel_size=[2, 2],
)

Example 2 — coded model with Conditioners: "2C2E1D"

model = build_model_from_code(
    "2C2E1D",
    input_shape=(1, 2048),
    latent_dim=128,
    num_classes=10,
    cond_dim=16,
    conditioner_sizes=[10, 32, 16],
    activation='relu',
    encoder_has_cnn=True,
    decoder_has_cnn=True,
    # ... plus the same feature lists as above
)
"""
