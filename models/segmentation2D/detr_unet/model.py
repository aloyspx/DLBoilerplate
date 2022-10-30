import torch
import torch.nn.functional as F
from positional_encodings.torch_encodings import PositionalEncoding2D, Summer
from torch import nn
from torchvision.models import resnet50

from models.segmentation2D.detr_unet.blocks.decoder_layer import DecoderLayer
from models.segmentation2D.detr_unet.blocks.encoder_layer import EncoderLayer


class DETRUnet(nn.Module):
    """
    Not TransUNet.
    Adds a transformer after bottleneck. No linear projection
    Added extra bells and whistles.
    """

    def __init__(self, in_channels: int = 3,
                 num_classes: int = 2,
                 nhead: int = 8,
                 num_enc: int = 6,
                 num_dec: int = 6,
                 d_model: int = 256):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Encoder
        ## Replace first layer
        self.encoder = EncoderLayer(in_channels, 64)
        self.backbone = resnet50(pretrained=True)

        del self.backbone.conv1
        del self.backbone.bn1
        del self.backbone.relu
        del self.backbone.maxpool
        del self.backbone.avgpool
        del self.backbone.fc

        self.channel_reduc = nn.Conv2d(2048, d_model, kernel_size=(1, 1))

        # Positional Encoding
        self.pos_enc = Summer(PositionalEncoding2D(d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.trans_enc = nn.TransformerEncoder(encoder_layer=enc_layer, num_layers=6)
        self.trans_enc.load_state_dict(torch.load("trans_init.pth"))

        # # Decoder
        self.decoder1 = DecoderLayer(d_model, 1024)
        self.decoder2 = DecoderLayer(1024, 512)
        self.decoder3 = DecoderLayer(512, 256)
        self.decoder4 = DecoderLayer(256, 64)

        self.final = nn.Conv2d(64, num_classes, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor):
        size = x.shape

        x, skip1 = self.encoder(x)
        x = self.backbone.layer1(x)
        skip2 = x
        x = self.backbone.layer2(x)
        skip3 = x
        x = self.backbone.layer3(x)
        skip4 = x
        x = self.backbone.layer4(x)

        x = self.channel_reduc(x)

        t_size = x.shape
        x = self.pos_enc(x).flatten(2).permute(2, 0, 1)
        x = self.trans_enc(x).permute(1, 2, 0).reshape(t_size)

        x = self.decoder1(x, skip4)
        x = self.decoder2(x, skip3)
        x = self.decoder3(x, skip2)
        x = self.decoder4(x, skip1)

        x = self.final(x)

        return F.interpolate(torch.sigmoid(x), size[-2:], mode="bilinear")


if __name__ == "__main__":
    net = DETRUnet()

    inp = torch.rand((1, 3, 512, 512))
    print(net(inp).shape)
