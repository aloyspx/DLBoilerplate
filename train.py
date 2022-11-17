import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from datasets.image2mask import Image2BinaryMask
## Train dataset
from models.medical.segmentation2D.unet.model import UNet
from utils.train_utils import train_one_epoch, evaluate

transform = A.Compose([
    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
    A.Flip(),
    ToTensorV2()
])

img_dir = "data/idrid_hard_exudate_segmentation512/img/train/"
mask_dir = "data/idrid_hard_exudate_segmentation512/grd/train/"
train_dataset = Image2BinaryMask(img_dir=img_dir, mask_dir=mask_dir, postfix_img="jpg", postfix_mask="tif",
                                 transform=transform, multichannel_label=False)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)

## Val dataset
img_dir = "data/idrid_hard_exudate_segmentation512/img/val/"
mask_dir = "data/idrid_hard_exudate_segmentation512/grd/val/"
val_dataset = Image2BinaryMask(img_dir=img_dir, mask_dir=mask_dir, postfix_img="jpg", postfix_mask="tif",
                               transform=transform, multichannel_label=True)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)

## Network initialization
net = UNet()

weight = torch.Tensor([1, 10])

if torch.cuda.is_available():
    net.to("cuda")
    weight = weight.to("cuda")

if torch.cuda.device_count() > 1:
    net = torch.nn.DataParallel(net)

## Optimizer, scheduler and loss function
optimizer = torch.optim.AdamW(net.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss(weight=weight)

## Hyperparameter
epochs = 500

for epoch in range(epochs):
    loss = train_one_epoch(net, train_dataloader, criterion, optimizer, scheduler=None, grad_clip=0.1, verbose=False)
    print(f"Loss for epoch [{epoch + 1}/{epochs}] : {loss}")
    val = evaluate(net, val_dataloader, metrics=["dice"])
    print(f"Dice score for epoch [{epoch + 1}/{epochs}] : {val['dice']}")

torch.save(net.state_dict(), "checkpoint.pth")
