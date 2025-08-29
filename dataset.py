# dataset.py
import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SketchDataset(Dataset):
    def __init__(self, folder, size=512):
        self.paths = sorted([p for p in glob(os.path.join(folder, "*")) if p.lower().endswith(("png","jpg","jpeg","webp","bmp"))])
        if len(self.paths) == 0:
            raise ValueError(f"No images found in {folder}")
        self.tf = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        return {"pixel_values": self.tf(img), "caption": None}
