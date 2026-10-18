import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.io import read_image

class DIV2KDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None, target_transform=None):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)

        self.transform = transform
        self.target_transform = target_transform
        self.lr_images = sorted(self.lr_dir.glob("*.png"))

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_path = self.lr_images[idx]
        hr_path = self.hr_dir / lr_path.name.replace("x2.png", ".png")

        
        lr = read_image(str(lr_path))
        hr = read_image(str(hr_path))

        
        transform = v2.ToDtype(torch.float32, scale=True)

        lr = transform(lr)
        hr = transform(hr)

        return lr, hr


training_data = DIV2KDataset(
    lr_dir="data/DIV2K_train_LR_bicubic/X2",
    hr_dir="data/DIV2K_train_HR"
)


validation_data = DIV2KDataset(
    lr_dir="data/DIV2K_valid_LR_bicubic/X2",
    hr_dir="data/DIV2K_valid_HR"
)