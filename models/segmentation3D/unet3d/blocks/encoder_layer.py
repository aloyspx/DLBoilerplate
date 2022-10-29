from typing import List

from torch import nn

from models.segmentation3D.unet3d.blocks.common import DoubleConv


class EncoderLayer(nn.Module):
    def __init__(self,
                 c1: int,
                 c2: int,
                 padding: int = 1,
                 activation: str = 'relu',
                 norm: str = 'batch',
                 input_shape: List = None,
                 pool: str = 'max',
                 pool_stride: int = 2):
        """
        Double convolution with pooling layer as described in U-Net paper. Added some bells and whistles options.
        :param c1: channels in
        :param c2: channels out
        :param activation: activation function -> relu or leaky relu
        :param norm: normalization -> batch, layer, group4, group8, group16
        :param input_shape: input shape, necessary for layer norm
        :param pool: pooling type -> max, avg
        :param pool_stride: pooling stride
        """
        super().__init__()

        self.conv = DoubleConv(c1, c2, padding, activation, norm, input_shape)

        assert pool in ['max', 'avg']

        ## Pooling
        if pool == 'max':
            self.pool = nn.MaxPool3d(2, stride=pool_stride)
        if pool == 'avg':
            self.pool = nn.AvgPool3d(2, stride=pool_stride)

    def forward(self, x):
        x = self.conv(x)
        return self.pool(x), x
