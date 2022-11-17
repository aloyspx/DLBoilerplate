### YOLO v1
from torch import nn


class YOLO(nn.Module):
    def __init__(self):
        super().__init__()
        # block 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, bias=False, kernel_size=(7, 7), stride=(2, 2), padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2))

        # block 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 192, bias=False, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2))

        # block 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(192, 128, bias=False, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, bias=False, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, bias=False, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, bias=False, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2)
        )

        # block 4
        self.layer4 = nn.Sequential(
            # 4 times
            nn.Conv2d(512, 256, bias=False, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, bias=False, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.Conv2d(512, 256, bias=False, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, bias=False, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.Conv2d(512, 256, bias=False, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, bias=False, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.Conv2d(512, 256, bias=False, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, bias=False, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            #
            nn.Conv2d(512, 512, bias=False, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, bias=False, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2)
        )

        # block 5
        self.layer5 = nn.Sequential(
            # 2 times
            nn.Conv2d(1024, 512, bias=False, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, bias=False, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, 512, bias=False, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, bias=False, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            #
            nn.Conv2d(1024, 1024, bias=False, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, bias=False, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1)
        )

        # block 6
        self.layer6 = nn.Sequential(
            nn.Conv2d(1024, 1024, bias=False, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, bias=False, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1)
        )

        # final mlp layer
        self.final = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, 7 * 7 * (90 + 2 * 5)),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.final(x)

        return x.reshape(-1, 7, 7, (90 + 2 * 5))
