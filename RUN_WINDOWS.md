# Hướng dẫn chạy trên Windows

Project nằm tại thư mục chứa file `app.py`, ví dụ:

```
C:\Users\HOAN\Downloads\student-behavior-recognition\student-behavior-recognition
```

## 1. Yêu cầu

| Thành phần | Khuyến nghị |
|------------|-------------|
| Python | **3.10** hoặc **3.12** (không dùng 3.13) |
| RAM | ≥ 16 GB |
| GPU | NVIDIA + CUDA (tùy chọn; không có GPU vẫn chạy được nhưng chậm) |
| Video demo | Đặt file `.mp4` vào thư mục `videos/` |

### Cài Python 3.12 (nếu máy đang là 3.13)

1. Tải Python 3.12 từ https://www.python.org/downloads/
2. Khi cài, bật **"Add python.exe to PATH"**
3. Kiểm tra:

```powershell
py -3.12 --version
```

## 2. Tạo môi trường ảo và cài thư viện

Mở **PowerShell**, vào thư mục project:

```powershell
cd C:\Users\HOAN\Downloads\student-behavior-recognition\student-behavior-recognition
```

Tạo và kích hoạt venv (dùng Python 3.12 nếu có):

```powershell
py -3.12 -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
```

### Cài PyTorch

**Có GPU NVIDIA (CUDA 12.1):**

```powershell
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

**Chỉ CPU (không có GPU):**

```powershell
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
```

### Cài các thư viện còn lại

```powershell
pip install -r requirements.txt
```

## 3. Kiểm tra import nhanh

```powershell
python -c "import app; print('app import ok')"
python -c "import train; print('train import ok')"
python -c "import evaluate; print('evaluate import ok')"
```

## 4. Chạy dashboard (demo chính)

```powershell
.\venv\Scripts\activate
python app.py
```

Mở trình duyệt: **http://localhost:5000**

1. Chọn video trong danh sách bên trái (từ thư mục `videos/`)
2. Chọn mô hình (mặc định: **M4 — Video Swin-T**)
3. Nhấn **Phân tích**

## 5. Demo CLI (xuất video có chú thích)

```powershell
python demo.py --input videos\D01_20240223064932.mp4 --output demo_out.mp4 --model 4
```

## 6. Huấn luyện / đánh giá (tùy chọn)

Cần dataset tại `dataset/{class}/*.mp4`. **Không dùng** `run_train.sh` trên Windows.

```powershell
python train.py --model 4
python evaluate.py --model 4
```

## 7. Xử lý lỗi thường gặp

| Lỗi | Cách xử lý |
|-----|------------|
| `No module named 'flask'` | Chạy lại `pip install -r requirements.txt` |
| `Checkpoint không tìm thấy` | Kiểm tra `cache\model4\best.pth` tồn tại |
| Video trống trên dashboard | Thêm file `.mp4` vào `videos/`, tên dạng `D01_20240223064932.mp4` |
| Chữ tiếng Việt bị ô vuông | Cài font Arial/Segoe (có sẵn trên Windows) |
| Rất chậm / hết RAM | Dùng model **M1** thay M4, hoặc bật GPU |
| Lần đầu chạy M3 chậm | Cần internet để tải weights HuggingFace |

## 8. Cấu hình demo (`config.py`)

| Biến | Mặc định | Ý nghĩa |
|------|----------|---------|
| `DEFAULT_MODEL_ID` | `4` | Model hành vi mặc định (Video Swin-T) |
| `YOLO_IMGSZ` | `1280` | Kích thước ảnh đầu vào YOLO |
| `BEHAVIOR_CLASSIFY_FPS_DIV` | `2` | Phân loại ~2 lần/giây (`fps // 2`) |
| `VIDEO_DIR` | `videos/` | Thư mục video lớp học |
