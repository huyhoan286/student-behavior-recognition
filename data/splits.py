import random
from pathlib import Path

from config import CLASSES, DATASET_ROOT, SPLIT_RATIOS, SPLIT_SEED, SPLITS_CACHE_PATH
from utils.cache import atomic_save_json, is_cached, load_json


def get_splits(force: bool = False) -> dict:
    """
    Trả về dict: {"train": [["video_id", "class"], ...], "val": [...], "test": [...]}
    Load từ cache nếu đã có, tạo mới nếu chưa.
    """
    if not force and is_cached(SPLITS_CACHE_PATH):
        print(f"[splits] Loaded from cache: {SPLITS_CACHE_PATH}")
        return load_json(SPLITS_CACHE_PATH)

    train_ratio, val_ratio, _ = SPLIT_RATIOS
    splits = {"train": [], "val": [], "test": []}

    rng = random.Random(SPLIT_SEED)

    for cls in CLASSES:
        cls_dir = DATASET_ROOT / cls
        videos = sorted([p.stem for p in cls_dir.glob("*.mp4")])
        rng.shuffle(videos)

        n = len(videos)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        splits["train"] += [[v, cls] for v in videos[:n_train]]
        splits["val"]   += [[v, cls] for v in videos[n_train:n_train + n_val]]
        splits["test"]  += [[v, cls] for v in videos[n_train + n_val:]]

    # shuffle để không bị sorted theo class
    for key in splits:
        rng.shuffle(splits[key])

    atomic_save_json(splits, SPLITS_CACHE_PATH)
    print(f"[splits] Created → train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    return splits
