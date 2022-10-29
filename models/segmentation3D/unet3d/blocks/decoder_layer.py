import torch
from torch import nn

from models.segmentation3D.unet3d.blocks.common import DoubleConv


class DecoderLayer(nn.Module):
    def __init__(self, c1: int, c2: int, padding: int = 1, concat_type: str = "pad",
                 activation='relu', norm='batch', input_shape=None):
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

        self.upsample = nn.ConvTranspose3d(c1, c1 // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2))

        ## Padding or cropping
        assert concat_type in ["pad", "crop"]
        self.concat_type = concat_type

        ## Convolutions
        self.conv = DoubleConv(c1, c2, padding, activation, norm, input_shape)

    def forward(self, x, skip):
        ## Upsampling step
        x = self.upsample(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
