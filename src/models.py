import torch.nn as nn
from torch import Tensor


class AutoEncoder(nn.Module):
    def __init__(self, encoder_config: dict, decoder_config: dict):
        """Initialize an AutoEncoder model.

        encoder_config (dict): Configuration dictionary for encoder layers.
            Keys are layer names (e.g., 'Linear', 'Conv2d') and values are
            dictionaries of layer parameters to pass to the layer constructor.
        decoder_config (dict): Configuration dictionary for decoder layers.
            Keys are layer names (e.g., 'Linear', 'Conv2d') and values are
            dictionaries of layer parameters to pass to the layer constructor.
        """
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential()
        for layer_name, layer_params in encoder_config.items():
            layer = getattr(nn, layer_name)(**layer_params)
            self.encoder.add_module(layer_name, layer)

        self.decoder = nn.Sequential()
        for layer_name, layer_params in decoder_config.items():
            layer = getattr(nn, layer_name)(**layer_params)
            self.decoder.add_module(layer_name, layer)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the autoencoder model.
        Encodes the input tensor through the encoder network, then decodes
        the encoded representation through the decoder network.
        Args:
            x (Tensor): Input tensor to be encoded and decoded.
        Returns:
            Tensor: Reconstructed output tensor after encoding and decoding.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def decode(self, x: Tensor) -> Tensor:
        """
        Decode only forward pass of the model.

        Args:
            x (Tensor): Input tensor to be decoded.

        Returns:
            Tensor: Reconstructed output tensor.
        """
        return self.decoder(x)


class DilatedAutoEncoder(AutoEncoder):
    NotImplementedError("DilatedAutoEncoder is not yet implemented.")
