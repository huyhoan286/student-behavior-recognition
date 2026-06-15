from pathlib import Path

ROOT          = Path(__file__).resolve().parent
DATASET_ROOT  = ROOT / "dataset"
CACHE_ROOT    = ROOT / "cache"

# Demo dashboard — video lớp học, YOLO, model mặc định
VIDEO_DIR          = ROOT / "videos"
THUMB_DIR          = ROOT / "static" / "thumbs"
YOLO_WEIGHTS       = ROOT / "yolov8s.pt"
DEFAULT_MODEL_ID   = 4
YOLO_IMGSZ         = 1280
# Phân loại hành vi mỗi (fps // BEHAVIOR_CLASSIFY_FPS_DIV) frame (~2 lần/giây với video 25fps)
BEHAVIOR_CLASSIFY_FPS_DIV = 2

CLASSES       = ["distracted", "drink_eat", "normal", "sleep", "use_smartphone"]
NUM_CLASSES   = len(CLASSES)
CLASS_TO_IDX  = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS  = {i: c for c, i in CLASS_TO_IDX.items()}

# Frame extraction
FRAMES_PER_VIDEO   = 30        # 1 frame/giây từ video 30 giây
FRAME_CACHE_DIR    = CACHE_ROOT / "frames"
SPLITS_CACHE_PATH  = CACHE_ROOT / "splits.json"
PLOTS_DIR          = CACHE_ROOT / "plots"

# Train/val/test split
SPLIT_RATIOS = (0.70, 0.15, 0.15)
SPLIT_SEED   = 42

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Kinetics-400 normalization (Video Swin pretrained stats)
KINETICS_MEAN = [0.4345, 0.4051, 0.3775]
KINETICS_STD  = [0.2768, 0.2713, 0.2737]

# Model 1 — EfficientNet-B0 frame classifier
M1 = dict(
    backbone       = "efficientnet_b0",
    img_size       = 224,
    mean           = IMAGENET_MEAN,
    std            = IMAGENET_STD,
    batch_size     = 64,
    lr             = 1e-4,
    epochs         = 30,
    freeze_epochs  = 5,
    dropout        = 0.4,
    checkpoint_dir = CACHE_ROOT / "model1",
)

# Model 3 — TimeSformer-base (Video Transformer, divided space-time attention)
M3 = dict(
    model_name     = "facebook/timesformer-base-finetuned-k400",
    num_frames     = 8,
    img_size       = 224,
    mean           = IMAGENET_MEAN,   # TimeSformer fine-tuned, dùng ImageNet stats
    std            = IMAGENET_STD,
    batch_size     = 4,               # 121M params — batch nhỏ để vừa VRAM
    lr             = 1e-5,            # fine-tune: lr nhỏ
    epochs         = 20,
    freeze_epochs  = 0,
    checkpoint_dir = CACHE_ROOT / "model3",
)

# Model 4 — Video Swin-T (Hierarchical Video Transformer)
M4 = dict(
    backbone       = "swin3d_t",
    num_frames     = 16,
    img_size       = 224,
    mean           = KINETICS_MEAN,
    std            = KINETICS_STD,
    dropout        = 0.3,
    batch_size     = 8,
    lr             = 1e-4,            # head lr
    backbone_lr    = 1e-5,            # encoder lr (differential LR)
    epochs         = 30,
    freeze_epochs  = 0,
    checkpoint_dir = CACHE_ROOT / "model4",
)
