import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

from config import CLASSES, DATASET_ROOT, FRAME_CACHE_DIR, FRAMES_PER_VIDEO


def _extract_one(video_path: Path, out_dir: Path, n_frames: int) -> str:
    sentinel = out_dir / ".done"
    if sentinel.exists():
        return f"skip:{video_path.stem}"

    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return f"error:{video_path.stem}:empty"

    indices = [int(total * i / n_frames) for i in range(n_frames)]

    for idx, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        out_file = out_dir / f"frame_{idx:03d}.jpg"
        cv2.imwrite(str(out_file), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

    cap.release()
    sentinel.touch()
    return f"done:{video_path.stem}"


def extract_all_frames(force: bool = False, workers: int = 8) -> None:
    tasks = []
    for cls in CLASSES:
        cls_dir = DATASET_ROOT / cls
        for video_path in sorted(cls_dir.glob("*.mp4")):
            out_dir = FRAME_CACHE_DIR / cls / video_path.stem
            if force and (out_dir / ".done").exists():
                (out_dir / ".done").unlink()
            tasks.append((video_path, out_dir))

    skipped = 0
    with tqdm(total=len(tasks), desc="Extracting frames", unit="video") as pbar:
        with ThreadPoolExecutor(max_workers=workers) as exe:
            futures = {exe.submit(_extract_one, vp, od, FRAMES_PER_VIDEO): vp for vp, od in tasks}
            for fut in as_completed(futures):
                result = fut.result()
                if result.startswith("skip"):
                    skipped += 1
                    tqdm.write(f"  cached: {result.split(':')[1]}")
                elif result.startswith("error"):
                    tqdm.write(f"  ERROR: {result}")
                pbar.update(1)

    print(f"[frames] Done. Skipped (cached): {skipped}/{len(tasks)}")


if __name__ == "__main__":
    extract_all_frames()
