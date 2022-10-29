import torch.nn.functional as F
from torch import nn


class WSConv3d(nn.Conv3d):
    """
    Paper: Micro-Batch Training with Batch-Channel Normalization and Weight Standardization
    Github: https://github.com/joe-siyuan-qiao/WeightStandardization
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                            keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class DoubleConv(nn.Module):
    def __init__(self, c1: int, c2: int, padding: int = 1, activation='relu', norm='batch', input_shape=None):
        """
        Double convolution as described in U-Net paper. Added some bells and whistles options.
        :param c1: channels in
        :param c2: channels out
        :param activation: activation function -> relu or leaky relu
        :param norm: normalization -> batch, layer, group4, group8, group16
        :param input_shape: input shape, necessary for layer norm
        """
        super().__init__()

        ## Activation
        assert activation in ['relu', 'leaky_relu']

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()

        ## Normalization
        assert norm in ['batch', 'layer', 'group_4', 'group_8', 'group_16']

        if norm == 'batch':
            self.norm1, self.norm2 = nn.BatchNorm3d(c2), nn.BatchNorm3d(c2)
        if norm == 'layer':
            assert input_shape
            h, w = input_shape[-2:]
            self.norm1, self.norm2 = nn.LayerNorm([c2, h - 2, w - 2]), nn.LayerNorm([c2, h - 4, w - 4])
        if norm == 'group_4':
            assert c2 % 4 == 0
            self.norm1, self.norm2 = nn.GroupNorm(4, c2), nn.GroupNorm(4, c2)
        if norm == 'group_8':
            assert c2 % 8 == 0
            self.norm1, self.norm2 = nn.GroupNorm(8, c2), nn.GroupNorm(8, c2)
        if norm == 'group_16':
            assert c2 % 16 == 0
            self.norm1, self.norm2 = nn.GroupNorm(16, c2), nn.GroupNorm(16, c2)

        ## Convolutions
        if "group" in norm:
            self.conv1 = WSConv3d(c1, c2, kernel_size=(3, 3, 3), padding=1)
            self.conv2 = WSConv3d(c2, c2, kernel_size=(3, 3, 3), padding=1)
        else:
            self.conv1 = nn.Conv3d(c1, c2, kernel_size=(3, 3, 3), padding=1)
            self.conv2 = nn.Conv3d(c2, c2, kernel_size=(3, 3, 3), padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return self.activation(x)
