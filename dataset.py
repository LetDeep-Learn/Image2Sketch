import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ImageCaptionDataset(Dataset):
    def __init__(self, data_dir, img_size=512, captions_txt=None, augment=False):
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, "images")
        if not os.path.isdir(self.img_dir):
            raise ValueError(f"images folder not found at {self.img_dir}")
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        if len(self.img_files) == 0:
            raise ValueError("No images found in dataset/images")
        self.captions = {}
        if captions_txt and os.path.isfile(captions_txt):
            with open(captions_txt, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    if "|" in line:
                        fname, cap = line.split("|", 1)
                        self.captions[fname.strip()] = cap.strip()
        self.img_size = img_size
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        fname = self.img_files[idx]
        path = os.path.join(self.img_dir, fname)
        with Image.open(path).convert("RGB") as im:
            img = im.copy()
        pixel_values = self.transform(img)
        caption = self.captions.get(fname, None)
        if caption is None:
            caption = os.path.splitext(fname)[0].replace("_"," ")
        return {"pixel_values": pixel_values, "caption": caption, "filename": fname}

def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    captions = [b["caption"] for b in batch]
    filenames = [b["filename"] for b in batch]
    return {"pixel_values": pixel_values, "captions": captions, "filenames": filenames}
