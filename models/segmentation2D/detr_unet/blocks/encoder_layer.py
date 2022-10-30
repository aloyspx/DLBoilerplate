from torch import nn

from models.segmentation2D.unet_transformer.blocks.common import DoubleConv


class EncoderLayer(nn.Module):
    def __init__(self, c1: int, c2: int):
        """
        Double convolution with pooling layer as described in U-Net paper. Added some bells and whistles options.
        :param c1: channels in
        :param c2: channels out
        :param activation: activation function -> relu or leaky relu
        :param norm: normalization -> batch, layer, group4, group8, group16
        :param input_shape: input shape, necessary for layer norm
        :param pool: pooling type -> max, avg
        """
        super().__init__()

        self.conv = DoubleConv(c1, c2)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        return self.pool(x), x
