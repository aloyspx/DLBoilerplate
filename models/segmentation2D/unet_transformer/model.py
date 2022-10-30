from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from positional_encodings.torch_encodings import PositionalEncoding2D, Summer
from torch import nn

from blocks.common import DoubleConv
from blocks.decoder_layer import DecoderLayer
from blocks.encoder_layer import EncoderLayer


class UNetTransformer(nn.Module):
    """
    Not TransUNet.
    Adds a transformer after bottleneck. No linear projection
    Added extra bells and whistles.
    """

    def __init__(self, in_channels: int = 3,
                 conv_channels: List = None,
                 num_classes: int = 2,
                 activation: str = 'relu',
                 norm: str = 'batch',
                 input_shape: List = None,
                 pool: str = 'max',
                 concat_type: str = 'pad',
                 nhead: int = 8,
                 num_layers: int = 6,
                 d_model: int = 1024):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Define number of output channels per conv
        if not conv_channels:
            self.conv_channels = [64, 128, 256, 512]
        else:
            assert (np.array(conv_channels) / conv_channels[0] == [1, 2, 4, 8]).all()
            self.conv_channels = conv_channels

        assert pool in ['max', 'avg']

        # Encoder
        self.encoder1 = EncoderLayer(self.in_channels, self.conv_channels[0], activation=activation,
                                     norm=norm, input_shape=input_shape, pool=pool)
        self.encoder2 = EncoderLayer(self.conv_channels[0], self.conv_channels[1], activation=activation,
                                     norm=norm, input_shape=input_shape, pool=pool)
        self.encoder3 = EncoderLayer(self.conv_channels[1], self.conv_channels[2], activation=activation,
                                     norm=norm, input_shape=input_shape, pool=pool)
        self.encoder4 = EncoderLayer(self.conv_channels[2], self.conv_channels[3], activation=activation,
                                     norm=norm, input_shape=input_shape, pool=pool)

        # Bottleneck
        self.bottleneck = DoubleConv(self.conv_channels[3], d_model, activation=activation,
                                     norm=norm, input_shape=input_shape)

        # Positional Encoding
        self.pos_enc = Summer(PositionalEncoding2D(d_model))

        # Transformer
        trans_enc_layer = nn.TransformerEncoderLayer(nhead=nhead, d_model=d_model)
        self.trans_enc = nn.TransformerEncoder(encoder_layer=trans_enc_layer, num_layers=num_layers)

        # Decoder
        self.decoder1 = DecoderLayer(d_model, self.conv_channels[3], concat_type=concat_type,
                                     activation=activation, norm=norm, input_shape=input_shape)
        self.decoder2 = DecoderLayer(self.conv_channels[3], self.conv_channels[2], concat_type=concat_type,
                                     activation=activation, norm=norm, input_shape=input_shape)
        self.decoder3 = DecoderLayer(self.conv_channels[2], self.conv_channels[1], concat_type=concat_type,
                                     activation=activation, norm=norm, input_shape=input_shape)
        self.decoder4 = DecoderLayer(self.conv_channels[1], self.conv_channels[0], concat_type=concat_type,
                                     activation=activation, norm=norm, input_shape=input_shape)

        self.final = nn.Conv2d(self.conv_channels[0], num_classes, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor):
        size = x.shape

        x, skip1 = self.encoder1(x)
        x, skip2 = self.encoder2(x)
        x, skip3 = self.encoder3(x)
        x, skip4 = self.encoder4(x)
        x = self.bottleneck(x)

        t_size = x.shape
        x = self.pos_enc(x)
        x = self.trans_enc(x.flatten(2).permute(2, 0, 1))
        x = x.permute(1, 2, 0).reshape(t_size)

        x = self.decoder1(x, skip4)
        x = self.decoder2(x, skip3)
        x = self.decoder3(x, skip2)
        x = self.decoder4(x, skip1)

        x = self.final(x)
        x = torch.sigmoid(x)
        return F.interpolate(x, size[-2:])


if __name__ == "__main__":
    net = UNetTransformer(concat_type="crop")

    inp = torch.rand((1, 3, 572, 572))
    print(net(inp).shape)
