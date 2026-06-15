import threading
import uuid
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from torchvision import transforms

from config import (
    BEHAVIOR_CLASSIFY_FPS_DIV,
    CACHE_ROOT,
    DEFAULT_MODEL_ID,
    IDX_TO_CLASS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    M1,
    M3,
    M4,
    ROOT,
    VIDEO_DIR,
    YOLO_IMGSZ,
    YOLO_WEIGHTS,
)

app = Flask(__name__, static_folder=str(ROOT / "static"), static_url_path="/static")
app.json.ensure_ascii = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAMES = {
    1: "EfficientNet-B0",
    3: "TimeSformer",
    4: "Video Swin-T",
}
CLASS_LABELS_VN = {
    "normal":         "Bình thường",
    "distracted":     "Mất tập trung",
    "sleep":          "Ngủ gật",
    "use_smartphone": "Dùng điện thoại",
    "drink_eat":      "Ăn uống",
}
CLASS_COLORS_RGB = {
    "normal":         (0, 200, 100),
    "distracted":     (59, 130, 246),
    "sleep":          (168, 85, 247),
    "use_smartphone": (245, 158, 11),
    "drink_eat":      (239, 104, 32),
}
CLASS_COLORS_BGR = {k: (v[2], v[1], v[0]) for k, v in CLASS_COLORS_RGB.items()}

_FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("C:/Windows/Fonts/segoeui.ttf"),
    Path("C:/Windows/Fonts/tahoma.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"),
]
_FONT_PATH = next((p for p in _FONT_CANDIDATES if p.exists()), None)
_font_cache: dict = {}

def _get_font(size: int):
    if size not in _font_cache:
        _font_cache[size] = (
            ImageFont.truetype(_FONT_PATH, size) if _FONT_PATH
            else ImageFont.load_default()
        )
    return _font_cache[size]


def _draw_label_pil(frame_bgr: np.ndarray, boxes_labels: list) -> np.ndarray:
    """Vẽ bounding box + text tiếng Việt lên frame dùng PIL."""
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font = _get_font(15)

    for x1, y1, x2, y2, tag, color_rgb in boxes_labels:
        draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=2)
        bbox = draw.textbbox((0, 0), tag, font=font)
        tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
        ty = max(y1 - th - 6, 0)
        draw.rectangle([x1, ty, x1 + tw + 8, ty + th + 6], fill=color_rgb)
        draw.text((x1 + 4, ty + 3), tag, font=font, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

ROOM_LABEL = {"D01": "Phòng 01", "D02": "Phòng 02", "D04": "Phòng 04"}

_model_cache: dict = {}
_stats_lock = threading.Lock()
_stream_session = {"id": None}
_current_stats  = {"active": False, "persons": {}, "overall_focus": 0.0, "frame": 0, "total": 0}


def _parse_videos():
    videos = []
    for mp4 in sorted(VIDEO_DIR.glob("*.mp4")):
        stem = mp4.stem
        parts = stem.split("_", 1)
        if len(parts) != 2:
            continue
        room, dt = parts[0], parts[1]
        try:
            time_str = f"{dt[8:10]}:{dt[10:12]}"
            date_str = f"{dt[6:8]}/{dt[4:6]}/{dt[:4]}"
        except Exception:
            time_str = stem; date_str = ""
        videos.append({
            "filename": mp4.name,
            "stem":     stem,
            "room":     room,
            "room_label": ROOM_LABEL.get(room, room),
            "time":     time_str,
            "date":     date_str,
            "label":    f"{ROOM_LABEL.get(room, room)} — {time_str} ({date_str})",
            "thumb":    f"/static/thumbs/{stem}.jpg",
            "path":     str(mp4),
        })
    return videos

VIDEOS = _parse_videos()


def _get_yolo():
    from ultralytics import YOLO
    return YOLO(str(YOLO_WEIGHTS))


def _load_behavior_model(model_id: int):
    configs = {1: M1, 3: M3, 4: M4}
    cfg = configs[model_id]
    ckpt_path = Path(cfg["checkpoint_dir"]) / "best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint không tìm thấy: {ckpt_path}")
    if model_id == 1:
        from models.model1 import Model1; model = Model1()
    elif model_id == 3:
        from models.model3 import Model3; model = Model3()
    else:
        from models.model4 import Model4; model = Model4()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model.to(DEVICE), cfg


def _get_behavior_model(model_id: int):
    if model_id not in _model_cache:
        _model_cache[model_id] = _load_behavior_model(model_id)
    return _model_cache[model_id]


def _build_transform(mean, std, img_size=224):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def _sample_uniform(buf: deque, n: int) -> list:
    frames = list(buf)
    total  = len(frames)
    return [frames[int(total * i / n)] for i in range(n)]


def generate_stream(video_path: str, model_id: int, session_id: str, start_sec: float = 0):
    behavior_model, cfg = _get_behavior_model(model_id)
    yolo      = _get_yolo()
    mean      = cfg.get("mean", IMAGENET_MEAN)
    std       = cfg.get("std",  IMAGENET_STD)
    seq_len   = cfg.get("num_frames", cfg.get("seq_len", 16))
    transform = _build_transform(mean, std)

    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)

    start_frame = int(start_sec * fps)
    # Tần suất phân loại hành vi: mỗi N frame (N ≈ fps/2 → ~2 lần/giây)
    classify_every = max(int(fps // BEHAVIOR_CLASSIFY_FPS_DIV), 1)

    # Temporal smoothing: chỉ tin kết quả khi đa số SMOOTH_WIN prediction gần nhất đồng ý
    # Min confidence: bỏ qua prediction có conf < MIN_CONF (coi như "normal")
    SMOOTH_WIN = 7    # cửa sổ N prediction gần nhất
    MIN_CONF   = 0.60 # bỏ qua nếu model không đủ tự tin

    person_buffers  = {}
    person_raw_hist = {}  # deque(maxlen=SMOOTH_WIN): lịch sử raw prediction
    person_labels   = {}  # nhãn đã smooth để hiển thị
    person_history  = {}  # toàn bộ lịch sử smooth (cho focus score)

    with _stats_lock:
        _current_stats.update({
            "active": True, "persons": {}, "overall_focus": 0.0,
            "frame": start_frame, "total": total,
        })

    try:
        frame_idx = start_frame
        while True:
            if _stream_session["id"] != session_id:
                break
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo.track(frame, persist=True, classes=[0],
                                 verbose=False, conf=0.20, iou=0.45, imgsz=YOLO_IMGSZ)

            active_ids = set()

            boxes_labels = []

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids   = results[0].boxes.id.cpu().numpy().astype(int)

                for box, tid in zip(boxes, ids):
                    x1 = max(0, int(box[0])); y1 = max(0, int(box[1]))
                    x2 = min(frame.shape[1], int(box[2])); y2 = min(frame.shape[0], int(box[3]))
                    active_ids.add(tid)

                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    if tid not in person_buffers:
                        person_buffers[tid]  = deque(maxlen=seq_len)
                        person_raw_hist[tid] = deque(maxlen=SMOOTH_WIN)
                        person_labels[tid]   = ("normal", 0.0)
                        person_history[tid]  = []

                    person_buffers[tid].append(
                        transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    )

                    buf = person_buffers[tid]
                    if len(buf) == seq_len and frame_idx % classify_every == 0:
                        sampled = _sample_uniform(buf, seq_len)
                        clip = torch.stack(sampled).unsqueeze(0).to(DEVICE)
                        if model_id == 1:
                            clip = clip[:, -1]
                        with torch.no_grad():
                            logits = behavior_model(clip)
                        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                        pred  = int(probs.argmax())
                        raw_label = IDX_TO_CLASS[pred]
                        raw_conf  = float(probs[pred])

                        # chỉ chấp nhận prediction nếu conf đủ cao
                        if raw_conf >= MIN_CONF:
                            person_raw_hist[tid].append(raw_label)
                        else:
                            person_raw_hist[tid].append("normal")  # nghi ngờ → coi là tập trung

                        # smoothed label = majority vote trong cửa sổ
                        window = list(person_raw_hist[tid])
                        smoothed = Counter(window).most_common(1)[0][0]
                        person_labels[tid] = (smoothed, raw_conf)
                        person_history[tid].append(smoothed)

                    label, conf = person_labels[tid]
                    color_rgb = CLASS_COLORS_RGB.get(label, (128, 128, 128))
                    tag = f"ID {tid}  {CLASS_LABELS_VN.get(label, label)}  {conf:.0%}"
                    boxes_labels.append((x1, y1, x2, y2, tag, color_rgb))

            frame = _draw_label_pil(frame, boxes_labels)

            if frame_idx % classify_every == 0:
                persons_stat = {}
                for tid in active_ids:
                    hist = person_history.get(tid, [])
                    lbl, conf = person_labels.get(tid, ("normal", 0.0))
                    focus = hist.count("normal") / len(hist) * 100 if hist else 0.0
                    persons_stat[str(tid)] = {
                        "label":    lbl,
                        "label_vn": CLASS_LABELS_VN.get(lbl, lbl),
                        "is_focus": lbl == "normal",
                        "conf":     round(conf * 100, 1),
                        "focus":    round(focus, 1),
                        "total":    len(hist),
                    }
                overall = (sum(p["focus"] for p in persons_stat.values()) / len(persons_stat)
                           if persons_stat else 0.0)
                with _stats_lock:
                    _current_stats.update({
                        "active": True, "persons": persons_stat,
                        "overall_focus": round(overall, 1),
                        "frame": frame_idx, "total": total,
                    })

            _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + jpg.tobytes() + b"\r\n")
            frame_idx += 1

    except GeneratorExit:
        pass
    finally:
        cap.release()
        with _stats_lock:
            _current_stats["active"] = False


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stream")
def stream():
    filename  = request.args.get("file", "")
    model_id  = int(request.args.get("model_id", DEFAULT_MODEL_ID))
    start_sec = float(request.args.get("start", 0))

    video_path = str(VIDEO_DIR / filename)
    if not Path(video_path).exists():
        return "Video không tồn tại", 404

    sid = uuid.uuid4().hex
    _stream_session["id"] = sid

    return Response(
        generate_stream(video_path, model_id, sid, start_sec),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/stream-stats")
def stream_stats():
    with _stats_lock:
        return jsonify(dict(_current_stats))


@app.route("/api/stop", methods=["POST"])
def stop():
    _stream_session["id"] = None
    return jsonify({"ok": True})


@app.route("/api/videos")
def get_videos():
    return jsonify(VIDEOS)


@app.route("/video/<path:filename>")
def serve_video(filename):
    return send_from_directory(str(VIDEO_DIR), filename, conditional=True)



@app.route("/cache/<path:filename>")
def serve_cache(filename):
    return send_from_directory(str(CACHE_ROOT), filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
