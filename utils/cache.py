import json
import numpy as np
from pathlib import Path


def is_cached(path: Path, mode: str = "file") -> bool:
    """
    mode='file' / 'npy' / 'json'  → file tồn tại và có nội dung
    mode='sentinel'                → .done sentinel tồn tại
    mode='dir'                     → thư mục tồn tại và không rỗng
    """
    if mode == "dir":
        return path.is_dir() and any(path.iterdir())
    if mode == "sentinel":
        return path.exists()
    return path.exists() and path.stat().st_size > 0


def atomic_save_npy(arr: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    tmp.rename(out_path)


def atomic_save_json(data: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.rename(out_path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())
