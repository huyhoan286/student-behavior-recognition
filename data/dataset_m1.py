from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import CLASS_TO_IDX, FRAME_CACHE_DIR, M1


def get_transforms(split: str):
    img_size = M1["img_size"]
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class FrameDataset(Dataset):
    """Mỗi item là 1 frame ảnh đơn lẻ → label."""

    def __init__(self, split_entries: list, split: str):
        self.transform = get_transforms(split)
        self.samples   = []
        for video_id, cls in split_entries:
            frame_dir = FRAME_CACHE_DIR / cls / video_id
            for p in sorted(frame_dir.glob("*.jpg")):
                self.samples.append((p, CLASS_TO_IDX[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label
