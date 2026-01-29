import torch.nn as nn
from torch import Tensor
import logging
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    (ROOT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(
        ROOT_DIR / "logs/models.log",
        encoding="utf-8",
        mode="w"
    )
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

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
        self.encoder = nn.ModuleList()
        for layer_config in encoder_config["layers"]:
            layer_name = layer_config["name"]
            layer_params = layer_config["params"]
            logger.debug(f"Encoder: Adding layer: {layer_name} with params: {layer_params}")
            layer = getattr(nn, layer_name)(**layer_params)
            self.encoder.append(layer)

        self.decoder = nn.ModuleList()
        for layer_config in decoder_config["layers"]:
            layer_name = layer_config["name"]
            layer_params = layer_config["params"]
            logger.debug(f"Decoder: Adding layer: {layer_name} with params: {layer_params}")
            layer = getattr(nn, layer_name)(**layer_params)
            self.decoder.append(layer)

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
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension if missing
        
        for layer in self.encoder:
            x = layer(x)
        
        for layer in self.decoder:
            x = layer(x)
        return x


class DilatedAutoEncoder(AutoEncoder):
    NotImplementedError("DilatedAutoEncoder is not yet implemented.")
