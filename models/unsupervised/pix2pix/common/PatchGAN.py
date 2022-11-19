from torch import nn


class PatchGAN(nn.Module):
    """PatchGAN as described in
    Precomputed real-time texture synthesis with markovian generative adversarial networks by Li et al."""

    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)
