import torch
from torch import nn

"""
Still need to rework this a bit.
Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks by Radford et al.
"""


class Generator(nn.Module):
    def __init__(self, latent_dim, out_channel=1, width=64, height=64):
        self.out_channel = out_channel
        self.width = width
        self.height = height
        self.latent_dim = latent_dim

        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 1024, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(128, out_channel, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(-1, self.latent_dim, 1, 1)
        x = self.up(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(),
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x.view(-1, 512))
        return torch.sigmoid(x)
