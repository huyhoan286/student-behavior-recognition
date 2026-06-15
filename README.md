# Hệ thống nhận diện hành vi sinh viên trong lớp học

Đồ án nghiên cứu ứng dụng học sâu và thị giác máy tính để nhận diện hành vi sinh viên từ video giám sát lớp học, phân tích mức độ tập trung và trực quan hóa kết quả qua dashboard thời gian thực.

---

## Tổng quan

Hệ thống nhận diện 5 hành vi đặc trưng của sinh viên trong lớp học:

| Nhãn | Mô tả |
|---|---|
| `normal` | Bình thường — đang chú ý học |
| `distracted` | Mất tập trung |
| `sleep` | Ngủ gật |
| `use_smartphone` | Dùng điện thoại |
| `drink_eat` | Ăn uống |

Kết quả phân loại được gộp thành 2 trạng thái hiển thị: **Tập trung** (`normal`) và **Mất tập trung** (4 hành vi còn lại).

---

## Kiến trúc 4 mô hình

| Mô hình | Kiến trúc | Đầu vào | Backbone pretrain | Accuracy | Macro F1 |
|---|---|---|---|---|---|
| **M1** | EfficientNet-B0 | 1 frame | ImageNet | 88.68% | 88.70% |
| **M2** | EfficientNet-B0 + BiLSTM | 30 frames | ImageNet | 92.45% | 92.37% |
| **M3** | TimeSformer-base | 8 frames | Kinetics-400 | 91.82% | 91.76% |
| **M4** | Video Swin-T | 16 frames | Kinetics-400 | **92.45%** | **92.47%** |

Các mô hình được xây dựng theo thứ tự từ cơ bản đến nâng cao, với tiêu chí mỗi mô hình sau tốt hơn mô hình trước về khả năng mô hình hóa thông tin thời gian.

---

## Cấu trúc thư mục

```
.
├── app.py                  # Flask dashboard (real-time streaming)
├── train.py                # Huấn luyện mô hình
├── evaluate.py             # Đánh giá và sinh biểu đồ so sánh
├── demo.py                 # Demo CLI: xuất video annotated
├── config.py               # Cấu hình toàn cục (model configs, paths)
├── run_train.sh            # Script huấn luyện tuần tự 4 mô hình
├── requirements.txt        # Dependencies Python
│
├── models/
│   ├── model1.py           # EfficientNet-B0 (frame-level)
│   ├── model2.py           # EfficientNet-B0 + BiLSTM
│   ├── model3.py           # TimeSformer (HuggingFace)
│   └── model4.py           # Video Swin-T (torchvision)
│
├── data/
│   ├── frame_extractor.py  # Trích xuất frame từ video gốc
│   ├── splits.py           # Chia tập train/val/test (70/15/15)
│   ├── dataset_m1.py       # Dataset frame-level (M1)
│   ├── dataset_m2.py       # Dataset clip-level với padding (M2)
│   └── dataset_video.py    # Dataset video-level uniform sampling (M3, M4)
│
├── training/
│   ├── trainer.py          # Vòng lặp huấn luyện chung
│   ├── metrics.py          # Tính toán metrics (Accuracy, F1, Kappa, MCC...)
│   └── callbacks.py        # Early stopping, checkpoint
│
├── templates/
│   └── index.html          # Dashboard UI (HTML/CSS/JS thuần)
│
├── cache/
│   ├── splits.json         # Phân chia tập dữ liệu đã lưu cache
│   ├── plots/              # Biểu đồ so sánh 4 mô hình
│   └── model{1-4}/         # Checkpoint, log, confusion matrix, report
│
└── MLIC-Edu/               # [CẦN ĐẶT THỦ CÔNG] Video lớp học thực tế
```

---

## Cài đặt

### Yêu cầu

- Python 3.10+
- CUDA 12.8+ (GPU Blackwell/RTX 50xx dùng PyTorch cu128; các GPU cũ hơn dùng PyTorch cu121)
- RAM ≥ 16GB, VRAM ≥ 8GB

### Cài thư viện

```bash
# Tạo virtual environment
python -m venv .venv && source .venv/bin/activate

# PyTorch với CUDA (chọn đúng phiên bản cho GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Các thư viện khác
pip install -r requirements.txt
pip install flask ultralytics
```

---

## Dữ liệu

### Dataset hành vi (để huấn luyện mô hình)

Dataset gồm 1041 video ngắn (~30 giây/video), mỗi video ghi lại 1 trong 5 hành vi. Đặt theo cấu trúc:

```
dataset/
├── normal/
│   ├── normal_001.mp4
│   ├── normal_002.mp4
│   └── ...
├── distracted/
├── drink_eat/
├── sleep/
└── use_smartphone/
```

Sau khi đặt dataset đúng vị trí, chạy trích xuất frame:

```bash
python -c "from data.frame_extractor import extract_all; extract_all()"
```

Frame được cache tại `cache/frames/` (tự động tạo).

### Video lớp học thực tế (cho dashboard)

Đặt các file video giám sát lớp học vào thư mục `MLIC-Edu/` ngay trong thư mục gốc dự án. Dashboard tự động nhận diện tất cả file `.mp4` trong thư mục này và hiển thị theo nhóm phòng (dựa vào prefix tên file `D01_`, `D02_`, ...).

```
MLIC-Edu/
├── D01_20240223064932.mp4
├── D01_20240223084534.mp4
├── D02_20240220080028.mp4
└── ...
```

Tên file theo định dạng: `{MãPhòng}_{YYYYMMDDHHMMSS}.mp4`

---

## Huấn luyện

```bash
# Huấn luyện toàn bộ 4 mô hình tuần tự
bash run_train.sh

# Hoặc huấn luyện từng mô hình
python train.py --model 1   # EfficientNet-B0
python train.py --model 2   # EfficientNet + BiLSTM
python train.py --model 3   # TimeSformer
python train.py --model 4   # Video Swin-T
```

Checkpoint tốt nhất được lưu tại `cache/model{N}/best.pth`.

---

## Đánh giá

```bash
# Đánh giá 1 mô hình
python evaluate.py --model 4

# So sánh tất cả 4 mô hình (sinh biểu đồ)
python evaluate.py --model 1 2 3 4
```

Kết quả lưu tại `cache/model{N}/classification_report.txt` và `cache/plots/`.

---

## Demo CLI

Xuất video có annotation (bounding box hành vi theo từng giây):

```bash
python demo.py --input video.mp4 --output out.mp4 --model 4
```

---

## Dashboard thời gian thực

```bash
python app.py
# Truy cập tại http://localhost:5000
```

### Tính năng dashboard

- **Chọn video** từ danh sách `MLIC-Edu/` (nhóm theo phòng), kèm thumbnail và thời gian quay
- **Seek** đến vị trí bất kỳ trong video (nhập số giây)
- **MJPEG streaming real-time**: phát và phân tích đồng thời, không cần chờ xử lý xong
- **YOLO person detection + tracking** (YOLOv8s): phát hiện và theo dõi từng người với ID cố định
- **Phân loại hành vi per person**: mỗi người được phân tích độc lập bằng mô hình đã chọn
- **Temporal smoothing**: giảm false positive bằng cách lấy majority vote trong cửa sổ 7 prediction (~3 giây), chỉ tin prediction có confidence ≥ 60%
- **Hiển thị 2 trạng thái**: Tập trung (xanh) / Mất tập trung (đỏ cam) với % lịch sử từng người

### Luồng xử lý

```
Frame video → YOLOv8s detect + track → Crop từng người
    → Buffer frames → Behavior model classify
    → Temporal smoothing → Hiển thị Tập trung / Mất tập trung
```

---

## Kết quả thực nghiệm (test set, 159 video)

| | M1 | M2 | M3 | M4 |
|---|---|---|---|---|
| Accuracy | 0.8868 | 0.9245 | 0.9182 | **0.9245** |
| Balanced Acc | 0.8873 | 0.9248 | 0.9187 | 0.9248 |
| Macro F1 | 0.8870 | 0.9237 | 0.9176 | **0.9247** |
| Weighted F1 | 0.8867 | 0.9235 | 0.9172 | 0.9245 |
| Cohen Kappa | 0.8585 | 0.9057 | 0.8978 | 0.9057 |
| MCC | 0.8617 | 0.9062 | 0.8981 | 0.9058 |
| Inf. ms/sample | **2.70** | 125.36 | 50.39 | 69.49 |

M4 (Video Swin-T) đạt độ chính xác cao nhất với F1 macro tốt nhất. M1 nhanh nhất (phù hợp thiết bị hạn chế tài nguyên).
