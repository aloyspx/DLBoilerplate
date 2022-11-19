import torch
from torch import nn


class EncoderLayer(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x), x


class DecoderLayer(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(c1, c2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x, skip):
        x = self.conv1(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv2(x)


class GAN_UNet(nn.Module):
    """
    Refactor this to use the U-Net in medical.segmentation2D folder.
    The stride of 2 instead of pooling makes it a bit of a mess.
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.enc1 = EncoderLayer(in_channels, 64)
        self.enc2 = EncoderLayer(64, 128)
        self.enc3 = EncoderLayer(128, 256)
        self.enc4 = EncoderLayer(256, 512)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1)
        )

        self.dec1 = DecoderLayer(1024, 512)
        self.dec2 = DecoderLayer(512, 256)
        self.dec3 = DecoderLayer(256, 128)
        self.dec4 = DecoderLayer(128, 64)

        self.final = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, self.num_classes, kernel_size=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec1(x, skip4)
        x = self.dec2(x, skip3)
        x = self.dec3(x, skip2)
        x = self.dec4(x, skip1)

        return self.final(x)
