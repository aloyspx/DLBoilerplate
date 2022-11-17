import torch
from torch import nn

"""
I'm not sure who exactly to credit but let's go with the first to propose the concept.
Generative Adversarial Networks by Goodfellow et al.
"""

class Generator(nn.Module):
    def __init__(self, latent_dim, out_channel=1, width=28, height=28):
        self.out_channel = out_channel
        self.width = width
        self.height = height

        super().__init__()
        self.lin = nn.Sequential(
            nn.Linear(latent_dim, 64 * width // 4 * height // 4),
            nn.LeakyReLU(0.1))
        self.up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, out_channel, kernel_size=(4, 4)),
        )

    def forward(self, x):
        x = self.lin(x)
        return self.up(x.view(-1, 64, self.width // 4, self.height // 4))


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, width=28, height=28):
        super().__init__()
        self.hidden_dim = 32 * (width - 4) * (height - 4)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(3, 3)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, 50),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x.view(-1, self.hidden_dim))
        return torch.sigmoid(x)
