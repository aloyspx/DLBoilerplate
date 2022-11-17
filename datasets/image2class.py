import glob

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


## TODO: test
class Image2Class(Dataset):

    def __init__(self, img_dir, postfix_img, class_file, transform=None):
        self.imgs = np.array(sorted(glob.glob(f"{img_dir}/*.{postfix_img}")))
        self.class_file = class_file
        if "csv" in self.class_file:
            self.lbls = pd.read_csv(class_file)
            self.rows = list(self.lbls.columns.values)
            assert len(self.rows) == 2
        else:
            raise NotImplementedError

        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.imgs[idx]
        img = np.array(Image.open(img_path))

        if "csv" in self.class_file:
            cls_rows = self.lbls.loc[self.lbls[self.rows[0]] == img_path.split('/')[-1]]
            if len(cls_rows) != 1:
                print(f"Zero or multi class result for this file: {img_path.split('/')[-1]}")
            cls = cls_rows[0][1]
        else:
            raise NotImplementedError

        if self.transform:
            transformed = self.transform(image=img)
            img = transformed["image"]

        return {'image': img, "class": cls}
