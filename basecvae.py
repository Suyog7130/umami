
from init import *
from utils import *


class BaseCoder(nn.Module):
    def __init__(self, **kwargs):
        super(BaseCoder, self).__init__()
        self.input_shape = kwargs.get('input_shape', None)
        self.latent_dim = kwargs.get('latent_dim', None)
        self.num_classes = kwargs.get('num_classes', None)

        self.has_pre_fc = kwargs.get('has_pre_fc', True)
        self.has_cnn = kwargs.get('has_cnn', True)
        self.has_post_fc = kwargs.get('has_post_fc', True)

        self.n_layers = kwargs.get('n_layers', None)
        self.n_layers_cnn = kwargs.get('n_layers_cnn', self.n_layers)
        self.n_layers_fc = kwargs.get('n_layers_fc', self.n_layers)
        self.n_layers_pre_fc = kwargs.get('n_layers_pre_fc', self.n_layers_fc)
        self.n_layers_pre_fc = kwargs.get('n_layers_pre_fc', self.n_layers_fc)
    
        self.pre_fc_in_features = kwargs.get('pre_fc_in_features', None)
        self.pre_fc_out_features = kwargs.get('pre_fc_out_features', None)
        self.cnn_in_channels = kwargs.get('cnn_in_channels', None)
        self.cnn_out_channels = kwargs.get('cnn_out_channels', None)
        self.cnn_kernel_size = kwargs.get('cnn_kernel_size', None)
        self.cnn_dilation = kwargs.get('cnn_dilation', None)
        self.post_fc_in_features = kwargs.get('post_fc_in_features', None)
        self.post_fc_out_features = kwargs.get('post_fc_out_features', None)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.batchnorm = nn.BatchNorm1d(self.input_shape[0])


    def linear_layer(self, in_features, out_features):
        """
        Creates a linear layer with ReLU activation.
        """
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU()
        )
    
    def conv_layer(self, in_channels, out_channels, kernel_size, dilation):
        """
        Creates a convolutional layer with ReLU activation and max pooling.
        """
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation),
            nn.MaxPool1d(kernel_size=kernel_size),
            nn.ReLU()
        )

    def fc(self, in_features, out_features):
        fc_seq = []
        assert len(in_features) == len(out_features)
        for i in range(self.n_layers_pre_fc):
            fc_seq.append(self.linear_layer(in_features[i], out_features[i]))
        return nn.Sequential(*fc_seq)
    
    def cnn(self, in_channels, out_channels, kernel_size, dilation):
        """ The parameters are lists of the same length as the number of layers! """
        cnn_seq = []
        assert len(in_channels) == len(out_channels) == len(kernel_size) == len(dilation)
        for i in range(self.n_layers_cnn):
            cnn_seq.append(self.conv_layer(in_channels[i], out_channels[i], kernel_size[i], dilation[i]))
        return nn.Sequential(*cnn_seq)


class BaseEncoder(BaseCoder):

    def __init__(self, **kwargs):
        super(BaseEncoder, self).__init__(**kwargs)
        self.concat_xy_before = kwargs.get('concat_xy_before', False)
    
    def forward(self, x, y):
        if self.concat_xy_before:
            x = torch.cat([x.view(x.size(0), -1), y], dim=1)
        else:
            x = x.view(x.size(0), -1)

        if self.has_pre_fc:
            x = self.fc(self.pre_fc_in_features, self.pre_fc_out_features)(x)

        if self.has_cnn:
            x = x.view(x.size(0), self.cnn_in_channels, -1)
            x = self.cnn(self.cnn_in_channels, self.cnn_out_channels, self.cnn_kernel_size, self.cnn_dilation)(x)
            x = x.view(x.size(0), -1)

        if not self.concat_xy_before:
            x = torch.cat([x.view(x.size(0), -1), y], dim=1)

        if self.has_post_fc:
            x = self.fc(self.post_fc_in_features, self.post_fc_out_features)(x)

        return x.view(-1, self.latent_dim)


class BaseDecoder(BaseCoder):
    def forward(self, z, y):
        assert z.size(1) == self.latent_dim
        assert y.size(1) == self.num_classes

        z = torch.cat([z, y], dim=1)

        if self.has_pre_fc:
            z = self.fc(self.pre_fc_in_features, self.pre_fc_out_features)(z)
        
        if self.has_cnn:
            z = z.view(z.size(0), self.cnn_in_channels, -1)
            z = self.cnn(self.cnn_in_channels, self.cnn_out_channels, self.cnn_kernel_size, self.cnn_dilation)(z)
            z = z.view(z.size(0), -1)

        if self.has_post_fc:
            z = self.fc(self.post_fc_in_features, self.post_fc_out_features)(z)

        return z.view(-1, *self.input_shape)


class BaseCVAE(nn.Module):
    """
    Base class for Conditional Variational Autoencoder (CVAE).
    This class is not intended to be used directly, but serves as a base
    for the CVAE implementation with variable number of Encoders and Decoders.
    The CNN and NN Fully Connected layers in each Encoder and Decoder are also
    optional and can be customized as per the requirements. The number of layers
    in the CNNs and NNs can be adjusted as needed as well.
    """
    def __init__(self, **kwargs):
        super(BaseCVAE, self).__init__()

        # Params for number of layers
        self.n_encoders = kwargs.get('n_encoders', 2)
        self.n_decoders = kwargs.get('n_decoders', 2)
        self.encoder_has_cnn = kwargs.get('encoder_has_cnn', True)
        self.decoder_has_cnn = kwargs.get('decoder_has_cnn', True)
        self.encoder_has_fc = kwargs.get('encoder_has_fc', True)
        self.decoder_has_fc = kwargs.get('decoder_has_fc', True)
        self.n_layers = kwargs.get('n_layers', 2)
        self.n_layers_cnn = kwargs.get('n_layers_cnn', self.n_layers)
        self.n_layers_fc = kwargs.get('n_layers_fc', self.n_layers)
        self.n_layers_cnn_encoder = kwargs.get('n_layers_cnn_encoder', self.n_layers_cnn)
        self.n_layers_cnn_decoder = kwargs.get('n_layers_cnn_decoder', self.n_layers_cnn)
        self.n_layers_fc_encoder = kwargs.get('n_layers_fc_encoder', self.n_layers_fc)
        self.n_layers_fc_decoder = kwargs.get('n_layers_fc_decoder', self.n_layers_fc)

        # Initialize encoders and decoders as ModuleLists
        self.encoders = nn.ModuleList([
            BaseEncoder(
            has_cnn=self.encoder_has_cnn,
            has_pre_fc=self.encoder_has_fc,
            has_post_fc=self.encoder_has_fc,
            n_layers=self.n_layers,
            n_layers_cnn=self.n_layers_cnn_encoder,
            n_layers_fc=self.n_layers_fc_encoder,
            **kwargs
            ) for _ in range(self.n_encoders)
        ])

        self.decoders = nn.ModuleList([
            BaseDecoder(
            has_cnn=self.decoder_has_cnn,
            has_pre_fc=self.decoder_has_fc,
            has_post_fc=self.decoder_has_fc,
            n_layers=self.n_layers,
            n_layers_cnn=self.n_layers_cnn_decoder,
            n_layers_fc=self.n_layers_fc_decoder,
            **kwargs
            ) for _ in range(self.n_decoders)
        ])

    def encode(self, x, y, encoder_idx=0):
        return self.encoders[encoder_idx](x, y)

    def decode(self, z, y, decoder_idx=0):
        return self.decoders[decoder_idx](z, y)

    def forward(self, x, y, encoder_idx=0, decoder_idx=0):
        z = self.encode(x, y, encoder_idx)
        out = self.decode(z, y, decoder_idx)
        return out
    

class TwoC2E1D(BaseCVAE):
    """
    A specific implementation of BaseCVAE with 2 encoders and 1 decoder.
    This class is a concrete example of how to use the BaseCVAE class.
    """
    def __init__(self, **kwargs):
        super(TwoC2E1D, self).__init__(n_encoders=2, n_decoders=1, **kwargs)

    def forward(self, x, y):
        z1 = self.encode(x, y, encoder_idx=0)
        z2 = self.encode(x, y, encoder_idx=1)
        z = torch.cat([z1, z2], dim=1)
        out = self.decode(z, y, decoder_idx=0)
        return out
    
    
class TwoC2E2D(BaseCVAE):
    """
    A specific implementation of BaseCVAE with 2 encoders and 2 decoders.
    This class is a concrete example of how to use the BaseCVAE class.
    """
    def __init__(self, **kwargs):
        super(TwoC2E2D, self).__init__(n_encoders=2, n_decoders=2, **kwargs)

    def forward(self, x, y):
        z1 = self.encode(x, y, encoder_idx=0)
        z2 = self.encode(x, y, encoder_idx=1)
        z = torch.cat([z1, z2], dim=1)
        out1 = self.decode(z, y, decoder_idx=0)
        out2 = self.decode(z, y, decoder_idx=1)
        return out1, out2
