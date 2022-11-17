import torch
from torch import nn

from models.medical.segmentation2D.transunet.blocks.common import DoubleConv


class DecoderLayer(nn.Module):
    def __init__(self, c1, c2, padding, activation='relu', norm='batch', input_shape=None):
        """
        Double convolution with pooling layer as described in U-Net paper. Added some bells and whistles options.
        :param c1: channels in
        :param c2: channels out
        :param padding: padding for conv
        :param activation: activation function -> relu or leaky relu
        :param norm: normalization -> batch, layer, group4, group8, group16
        :param input_shape: input shape, necessary for layer norm
        """
        super().__init__()

        ## Convolutions
        self.conv = DoubleConv(c1, c2, padding, activation, norm, input_shape)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([skip, x], dim=1)
        return self.conv(x)
