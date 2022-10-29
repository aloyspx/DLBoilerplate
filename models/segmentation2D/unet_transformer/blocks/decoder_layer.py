import math

import torch
import torch.nn.functional as F
from torch import nn

from models.segmentation2D.unet_transformer.blocks.common import DoubleConv


class DecoderLayer(nn.Module):
    def __init__(self, c1: int, c2: int, concat_type: str = "pad",
                 activation='relu', norm='batch', input_shape=None):
        """
        Double convolution with pooling layer as described in U-Net paper. Added some bells and whistles options.
        :param c1: channels in
        :param c2: channels out
        :param activation: activation function -> relu or leaky relu
        :param norm: normalization -> batch, layer, group4, group8, group16
        :param input_shape: input shape, necessary for layer norm
        """
        super().__init__()

        self.upsample = nn.ConvTranspose2d(c1, c1 // 2, kernel_size=(2, 2), stride=(2, 2))

        ## Padding or cropping
        assert concat_type in ["pad", "crop"]
        self.concat_type = concat_type

        ## Convolutions
        self.conv = DoubleConv(c1, c2, activation, norm, input_shape)

    def forward(self, x, skip):
        ## Upsampling step
        x = self.upsample(x)

        ## Concat step
        if self.concat_type == "pad":
            x1, y1 = skip.shape[-2:]
            x2, y2 = x.shape[-2:]
            x = F.pad(x, (math.floor((y1 - y2) / 2), math.ceil((y1 - y2) / 2),
                          math.floor((x1 - x2) / 2), math.ceil((x1 - x2) / 2)),
                      mode="constant", value=0)

        if self.concat_type == "crop":
            y1, x1 = skip.shape[-2:]
            y2, x2 = x.shape[-2:]
            startx = x1 // 2 - (y2 // 2)
            starty = y1 // 2 - (y2 // 2)
            skip = skip[:, :, starty:starty + y2, startx:startx + x2]

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
