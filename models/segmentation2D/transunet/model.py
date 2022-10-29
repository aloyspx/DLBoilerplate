from typing import List

import numpy as np
import torch
from einops.layers.torch import Rearrange
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from torch import nn

from blocks.common import SingleConv
from blocks.decoder_layer import DecoderLayer
from blocks.encoder_layer import EncoderLayer


class UNetTransformer(nn.Module):
    """
    TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation
    Adds a transformer after bottleneck. With linear projection
    Added extra bells and whistles.
    """

    def __init__(self, in_channels: int = 3,
                 conv_channels: List = None,
                 num_classes: int = 2,
                 activation: str = 'relu',
                 norm: str = 'batch',
                 input_shape: List = None,
                 pool: str = 'max',
                 nhead: int = 8,
                 num_layers: int = 6,
                 patch_size: int = 16,
                 d_model: int = 512):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.d_model = d_model

        # Define number of output channels per conv
        if not conv_channels:
            self.conv_channels = [64, 128, 256]
        else:
            assert (np.array(conv_channels) / conv_channels[0] == [1, 2, 4]).all()
            self.conv_channels = conv_channels

        # Encoder
        self.encoder1 = EncoderLayer(self.in_channels, self.conv_channels[0], padding=1,
                                     activation=activation, norm=norm, input_shape=input_shape, pool=pool)
        self.encoder2 = EncoderLayer(self.conv_channels[0], self.conv_channels[1], padding=1,
                                     activation=activation, norm=norm, input_shape=input_shape, pool=pool)
        self.encoder3 = EncoderLayer(self.conv_channels[1], self.conv_channels[2], padding=1,
                                     activation=activation, norm=norm, input_shape=input_shape, pool=pool)

        # Linear Projection
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=16, p2=16),
            nn.Linear(patch_size * patch_size * in_channels, d_model),
        )

        # Positional Encoding
        self.pos_enc = Summer(PositionalEncoding1D(d_model))

        # Transformer
        trans_enc_layer = nn.TransformerEncoderLayer(nhead=nhead, d_model=d_model)
        self.trans_enc = nn.TransformerEncoder(encoder_layer=trans_enc_layer, num_layers=num_layers)

        # Bottleneck
        self.bottleneck = SingleConv(d_model, d_model // 2, padding=1, activation=activation, norm=norm,
                                     input_shape=input_shape)

        # Decoder
        self.decoder1 = DecoderLayer(d_model, self.conv_channels[2] // 2, padding=1,
                                     activation=activation, norm=norm, input_shape=input_shape)
        self.decoder2 = DecoderLayer(self.conv_channels[2], self.conv_channels[1] // 2, padding=1,
                                     activation=activation, norm=norm, input_shape=input_shape)
        self.decoder3 = DecoderLayer(self.conv_channels[1], self.conv_channels[0], padding=1,
                                     activation=activation, norm=norm, input_shape=input_shape)

        self.final = DecoderLayer(self.conv_channels[0], num_classes, padding=1, activation=activation,
                                  norm=norm, input_shape=input_shape)

    def forward(self, x: torch.Tensor):

        # Encoder
        x1 = self.encoder1(x)
        skip1 = x1
        x1 = self.encoder2(x1)
        skip2 = x1
        x1 = self.encoder3(x1)
        skip3 = x1

        ## Transformer
        h, w = x.shape[-2] // 16, x.shape[-1] // 16
        x2 = self.pos_enc(self.to_patch_embedding(x))
        x2 = self.trans_enc(x2.permute(1, 0, 2))
        x2 = self.bottleneck(x2.permute(1, 2, 0).reshape(-1, self.d_model, h, w))

        # Decoder
        y = self.decoder1(x2, skip3)
        y = self.decoder2(y, skip2)
        y = self.decoder3(y, skip1)
        y = self.final(y)

        return y


if __name__ == "__main__":
    net = UNetTransformer()
    inp = torch.rand((1, 3, 224, 224))
    print(net(inp).shape)
