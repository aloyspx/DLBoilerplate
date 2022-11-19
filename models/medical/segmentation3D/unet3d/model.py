from typing import List

import numpy as np
import torch
from torch import nn

from blocks.common import DoubleConv
from blocks.decoder_layer import DecoderLayer
from blocks.encoder_layer import EncoderLayer


class UNet3D(nn.Module):
    """
    3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. Cicek et al.
    3D version of U-Net. Not as many encoder/decoder layers to avoid high memory consumption.
    Added extra bells and whistles.
    """

    def __init__(self,
                 in_channels: int = 1,
                 conv_channels: List = None,
                 padding: int = 1,
                 num_classes: int = 2,
                 activation: str = 'relu',
                 norm: str = 'batch',
                 input_shape: List = None,
                 pool: str = 'max',
                 pool_stride: int = 2,
                 sigmoid: bool = True
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.sigmoid = sigmoid

        # Define number of output channels per conv
        if not conv_channels:
            self.conv_channels = [64, 128, 256, 512]
        else:
            assert (np.array(conv_channels) / conv_channels[0] == [1, 2, 4, 8]).all()
            self.conv_channels = conv_channels

        # Encoder
        self.encoder1 = EncoderLayer(self.in_channels, self.conv_channels[0], padding=padding, activation=activation,
                                     norm=norm, input_shape=input_shape, pool=pool, pool_stride=pool_stride)
        self.encoder2 = EncoderLayer(self.conv_channels[0], self.conv_channels[1], padding=padding,
                                     activation=activation,
                                     norm=norm, input_shape=input_shape, pool=pool, pool_stride=pool_stride)
        self.encoder3 = EncoderLayer(self.conv_channels[1], self.conv_channels[2], padding=padding,
                                     activation=activation,
                                     norm=norm, input_shape=input_shape, pool=pool, pool_stride=pool_stride)

        # Bottleneck
        self.bottleneck = DoubleConv(self.conv_channels[2], self.conv_channels[3], padding=padding,
                                     activation=activation, norm=norm, input_shape=input_shape)

        # Decoder
        self.decoder1 = DecoderLayer(self.conv_channels[3], self.conv_channels[2], padding=padding,
                                     activation=activation, norm=norm, input_shape=input_shape)
        self.decoder2 = DecoderLayer(self.conv_channels[2], self.conv_channels[1], padding=padding,
                                     activation=activation, norm=norm, input_shape=input_shape)
        self.decoder3 = DecoderLayer(self.conv_channels[1], self.conv_channels[0], padding=padding,
                                     activation=activation, norm=norm, input_shape=input_shape)

        self.final = nn.Conv3d(self.conv_channels[0], num_classes, kernel_size=(1, 1, 1))

    def forward(self, x: torch.Tensor):
        x, skip1 = self.encoder1(x)
        x, skip2 = self.encoder2(x)
        x, skip3 = self.encoder3(x)

        x = self.bottleneck(x)

        x = self.decoder1(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder3(x, skip1)

        x = self.final(x)

        if self.sigmoid:
            x = torch.sigmoid(x)

        return x


if __name__ == "__main__":
    net = UNet3D()

    inp = torch.rand((1, 1, 128, 128, 128))
    print(net(inp).shape)
