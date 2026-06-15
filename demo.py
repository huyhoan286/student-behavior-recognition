"""
Demo inference trên video: nhận video đầu vào, xuất video annotated.
Usage:
  python demo.py --input path/to/video.mp4 --output out.mp4 --model 4
  python demo.py --input video.mp4 --model 3   # TimeSformer
  python demo.py --input video.mp4 --model 4   # Video Swin-T (default)
"""
import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from config import CLASSES, IDX_TO_CLASS, IMAGENET_MEAN, IMAGENET_STD, KINETICS_MEAN, KINETICS_STD, M1, M3, M4

FOCUS_CLASSES = {"normal"}
CLASS_COLORS  = {
    "normal":         (0, 200, 0),
    "distracted":     (0, 100, 220),
    "sleep":          (150, 0, 220),
    "use_smartphone": (0, 165, 255),
}


def load_model(model_id: int):
    configs = {1: M1, 3: M3, 4: M4}
    cfg       = configs[model_id]
    ckpt_path = Path(cfg["checkpoint_dir"]) / "best.pth"
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint không tìm thấy: {ckpt_path}\nHãy train model {model_id} trước.")

    if model_id == 1:
        from models.model1 import Model1; model = Model1()
    elif model_id == 3:
        from models.model3 import Model3; model = Model3()
    else:
        from models.model4 import Model4; model = Model4()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def build_transform(mean: list, std: list, img_size: int = 224):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def draw_label(frame, label: str, conf: float):
    color = CLASS_COLORS.get(label, (255, 255, 255))
    text  = f"{label}  {conf:.0%}" if conf > 0 else label
    cv2.rectangle(frame, (0, 0), (370, 46), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2, cv2.LINE_AA)


def draw_timeline(frame, history: list):
    h, w   = frame.shape[:2]
    bar_h  = 22
    bar_y  = h - bar_h - 2
    n      = max(len(history), 1)
    seg_w  = w / n

    for i, cls in enumerate(history):
        x1 = int(i * seg_w); x2 = int((i + 1) * seg_w)
        color = CLASS_COLORS.get(cls, (128, 128, 128))
        cv2.rectangle(frame, (x1, bar_y), (x2, h - 2), color, -1)

    cv2.putText(frame, "timeline", (4, bar_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


def run_demo(input_path: str, output_path: str, model_id: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(model_id)
    model = model.to(device)

    mean = cfg.get("mean", IMAGENET_MEAN)
    std  = cfg.get("std",  IMAGENET_STD)
    seq_len  = cfg.get("num_frames", cfg.get("seq_len", 16))
    img_size = cfg.get("img_size", 224)
    transform = build_transform(mean, std, img_size)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        sys.exit(f"Không mở được video: {input_path}")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    buffer    = deque(maxlen=seq_len)
    history   = []
    cur_label = "..."
    cur_conf  = 0.0
    step      = max(1, int(fps))   # chạy inference mỗi giây

    print(f"Model {model_id} | seq_len={seq_len} | {total} frames @ {fps:.0f}fps")

    for frame_idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        # buffer frame đã transform (RGB tensor)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buffer.append(transform(rgb))

        # inference mỗi `step` frames khi buffer đủ
        if frame_idx % step == 0 and len(buffer) == seq_len:
            clip = torch.stack(list(buffer)).unsqueeze(0).to(device)
            # clip: [1, T, C, H, W] — đúng format cho cả 4 model
            # M1 cần [B, C, H, W] → dùng frame cuối
            if model_id == 1:
                clip = clip[:, -1]   # [1, C, H, W]

            with torch.no_grad():
                logits = model(clip)
            probs     = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx  = probs.argmax()
            cur_label = IDX_TO_CLASS[int(pred_idx)]
            cur_conf  = float(probs[pred_idx])
            history.append(cur_label)

        draw_label(frame, cur_label, cur_conf)
        draw_timeline(frame, history)
        out.write(frame)

        if frame_idx % 300 == 0:
            print(f"  {frame_idx}/{total}  current: {cur_label}")

    cap.release()
    out.release()
    print(f"Demo saved → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Input video path")
    parser.add_argument("--output", default="demo_out.mp4", help="Output video path")
    parser.add_argument("--model",  type=int, default=4, choices=[1, 3, 4],
                        help="1=CNN  3=TimeSformer  4=VideoSwin-T")
    args = parser.parse_args()
    run_demo(args.input, args.output, args.model)

if __name__ == "__main__":
    main()


