import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class Image2Mask(Dataset):

    def __init__(self, img_dir, postfix_img, mask_dir, postfix_mask, transform=None):
        self.imgs = np.array(sorted(glob.glob(f"{img_dir}/*.{postfix_img}")))
        self.masks = np.array(sorted(glob.glob(f"{mask_dir}/*.{postfix_mask}")))
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.imgs[idx]
        mask_path = self.masks[idx]

        img = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))

        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        return {'image': img, "mask": mask}
