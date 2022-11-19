import glob

from PIL import Image
from torch.utils.data import Dataset


class CelebA2GANImage(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = sorted(glob.glob(root_dir + "/*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.root_dir)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.root_dir[idx]))
