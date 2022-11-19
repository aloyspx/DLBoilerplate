import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class Image2Mask(Dataset):

    def __init__(self, img_dir, postfix_img, mask_dir, postfix_mask, num_cls, transform=None, target_transform=None):
        self.imgs = np.array(sorted(glob.glob(f"{img_dir}/*.{postfix_img}")))
        self.masks = np.array(sorted(glob.glob(f"{mask_dir}/*.{postfix_mask}")))
        self.transform = transform
        self.target_transform = target_transform
        self.num_cls = num_cls

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.imgs[idx]
        mask_path = self.masks[idx]

        img = Image.open(img_path)
        label = np.array(Image.open(mask_path))

        classes = np.unique(label)

        mask = np.zeros((*label.shape, self.num_cls))

        for cls in classes:
            mask[label == cls, cls - 1] = 1

        img = self.transform(img)
        mask = self.target_transform(mask).type(torch.float)

        return img, mask
