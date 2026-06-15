import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import CLASS_TO_IDX, FRAME_CACHE_DIR


def _build_transform(split: str, img_size: int, mean: list, std: list):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size + 16, img_size + 16)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def _sample_uniform(frames: list, n: int) -> list:
    """Lấy n frame phân bố đều từ danh sách frames."""
    total = len(frames)
    if total == 0:
        return []
    indices = [int(total * i / n) for i in range(n)]
    return [frames[min(i, total - 1)] for i in indices]


class VideoClipDataset(Dataset):
    """
    Dataset dùng chung cho M3 (TimeSformer) và M4 (Video Swin-T).
    Mỗi item: (clip [T, C, H, W], label)
    T = seq_len được lấy đều từ FRAMES_PER_VIDEO frames đã extract.
    """

    def __init__(self, split_entries: list, split: str,
                 seq_len: int, mean: list, std: list, img_size: int = 224):
        self.seq_len   = seq_len
        self.transform = _build_transform(split, img_size, mean, std)
        self.clips     = []

        for video_id, cls in split_entries:
            frame_dir = FRAME_CACHE_DIR / cls / video_id
            frames    = sorted(frame_dir.glob("*.jpg"))
            if frames:
                self.clips.append((frames, CLASS_TO_IDX[cls]))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        frames, label = self.clips[idx]
        sampled = _sample_uniform(frames, self.seq_len)
        imgs    = [self.transform(Image.open(f).convert("RGB")) for f in sampled]
        return torch.stack(imgs), label   # [T, C, H, W]
