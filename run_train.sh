#!/bin/bash
# Train 4 models song song trên 2 GPU.
# Session tồn tại kể cả terminal/VSCode đóng.
#
# Chạy:  bash run_train.sh
# Theo dõi: tmux attach -t train4
# Tắt hết:  tmux kill-session -t train4

set -e

SESSION="train3"
WORKSPACE="/workspace"
LOG_DIR="$WORKSPACE/logs"
mkdir -p "$LOG_DIR"

# Xoá session cũ nếu có
tmux kill-session -t "$SESSION" 2>/dev/null && echo "Killed old session '$SESSION'" || true

# ── GPU 0: M1 (EfficientNet-B0, ~2GB) ───────────────────────────────────────
tmux new-session  -d -s "$SESSION" -n "M1-EfficientNet" \
    "CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$WORKSPACE \
     python3 $WORKSPACE/train.py --model 1 \
     2>&1 | tee $LOG_DIR/model1.log; echo '[M1 DONE]'; read"

# ── GPU 0: M3 (TimeSformer, ~4.5GB) — GPU 0 tổng ~6.5GB, an toàn ───────────
tmux new-window   -t "$SESSION" -n "M3-TimeSformer" \
    "CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$WORKSPACE \
     python3 $WORKSPACE/train.py --model 3 \
     2>&1 | tee $LOG_DIR/model3.log; echo '[M3 DONE]'; read"

# ── GPU 1: M4 (Video Swin-T, ~15GB) — cần nguyên 1 GPU ─────────────────────
tmux new-window   -t "$SESSION" -n "M4-VideoSwin" \
    "CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$WORKSPACE \
     python3 $WORKSPACE/train.py --model 4 \
     2>&1 | tee $LOG_DIR/model4.log; echo '[M4 DONE]'; read"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  3 models đang train trong tmux session '$SESSION'   ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Re-attach :  tmux attach -t $SESSION                ║"
echo "║  Đổi window:  Ctrl+B  rồi  0 / 1 / 2                ║"
echo "║  Detach    :  Ctrl+B  rồi  d                         ║"
echo "║  Tắt hết   :  tmux kill-session -t $SESSION          ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Xem log (không cần attach):                         ║"
echo "║    tail -f logs/model3.log                           ║"
echo "║    tail -f logs/model1.log                           ║"
echo "║    tail -f logs/model4.log                           ║"
echo "║  VRAM:  watch -n2 nvidia-smi                         ║"
echo "╚══════════════════════════════════════════════════════╝"
