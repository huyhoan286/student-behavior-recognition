"""
Đánh giá 1 hoặc nhiều model trên test split và tạo biểu đồ so sánh.
Usage:
  python evaluate.py --model 1            # đánh giá riêng model 1
  python evaluate.py --model 1 2 3 4      # so sánh cả 4 model
"""
import argparse
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from config import (CLASS_TO_IDX, CLASSES, FRAME_CACHE_DIR, IDX_TO_CLASS,
                    M1, M3, M4, NUM_CLASSES, PLOTS_DIR)
from data.splits import get_splits
from training.metrics import compute_metrics

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────────

def load_model(model_id: int):
    configs = {1: M1, 3: M3, 4: M4}
    cfg = configs[model_id]
    ckpt_path = Path(cfg["checkpoint_dir"]) / "best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint for model {model_id}: {ckpt_path}")

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


@torch.no_grad()
def evaluate_model(model_id: int, splits: dict) -> dict:
    model, cfg = load_model(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)

    if model_id == 1:
        from data.dataset_m1 import FrameDataset
        ds = FrameDataset(splits["test"], split="test")
        loader = DataLoader(ds, batch_size=cfg["batch_size"] * 2,
                            shuffle=False, num_workers=4)
    else:
        # M3 (TimeSformer) và M4 (Video Swin) dùng chung VideoClipDataset
        from data.dataset_video import VideoClipDataset
        ds = VideoClipDataset(splits["test"], "test",
                              cfg["num_frames"], cfg["mean"], cfg["std"])
        loader = DataLoader(ds, batch_size=cfg["batch_size"],
                            shuffle=False, num_workers=2)

    all_preds, all_labels = [], []
    t0 = time.time()
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        all_preds  += logits.argmax(1).cpu().tolist()
        all_labels += y.tolist()
    elapsed = time.time() - t0

    n_samples = len(all_labels)
    # đối với model 1 ta dùng majority vote theo video
    if model_id == 1:
        all_preds, all_labels = _frame_to_video_vote(
            all_preds, all_labels, splits["test"])

    metrics = compute_metrics(all_preds, all_labels, CLASSES)
    metrics["inference_ms"] = elapsed * 1000 / max(n_samples, 1)

    # lưu report
    ckpt_dir = Path(cfg["checkpoint_dir"])
    extra = (
        f"\n--- Aggregate Metrics ---\n"
        f"Accuracy          : {metrics['accuracy']:.4f}\n"
        f"Balanced Accuracy : {metrics['balanced_accuracy']:.4f}\n"
        f"Macro F1          : {metrics['macro_f1']:.4f}\n"
        f"Weighted F1       : {metrics['weighted_f1']:.4f}\n"
        f"Macro Precision   : {metrics['macro_precision']:.4f}\n"
        f"Macro Recall      : {metrics['macro_recall']:.4f}\n"
        f"Cohen's Kappa     : {metrics['cohen_kappa']:.4f}\n"
        f"MCC               : {metrics['mcc']:.4f}\n"
        f"Inference ms/sample: {metrics['inference_ms']:.4f}\n"
    )
    (ckpt_dir / "classification_report.txt").write_text(metrics["report"] + extra)
    _save_confusion_matrix(metrics["confusion_matrix"], CLASSES,
                           ckpt_dir / "confusion_matrix.png",
                           title=f"Model {model_id} — Confusion Matrix")
    print(f"\n=== Model {model_id} ===")
    print(metrics["report"])
    print(
        f"  Accuracy          : {metrics['accuracy']:.4f}\n"
        f"  Balanced Accuracy : {metrics['balanced_accuracy']:.4f}\n"
        f"  Macro F1          : {metrics['macro_f1']:.4f}\n"
        f"  Weighted F1       : {metrics['weighted_f1']:.4f}\n"
        f"  Macro Precision   : {metrics['macro_precision']:.4f}\n"
        f"  Macro Recall      : {metrics['macro_recall']:.4f}\n"
        f"  Cohen's Kappa     : {metrics['cohen_kappa']:.4f}\n"
        f"  MCC               : {metrics['mcc']:.4f}\n"
        f"  Inference (ms/sample): {metrics['inference_ms']:.4f}"
    )
    return metrics


def _frame_to_video_vote(preds, labels, test_entries):
    """Majority vote: gom tất cả frame của cùng 1 video, vote nhãn.
    Dùng số frame thực tế (không hardcode) để tránh lỗi nếu video thiếu frame.
    """
    video_preds, video_labels = [], []
    ptr = 0
    for vid_id, cls in test_entries:
        frame_dir = FRAME_CACHE_DIR / cls / vid_id
        n_frames  = len(list(frame_dir.glob("*.jpg")))
        if n_frames == 0 or ptr >= len(preds):
            continue
        chunk = preds[ptr: ptr + n_frames]
        ptr  += n_frames
        vote  = max(set(chunk), key=chunk.count)
        video_preds.append(vote)
        video_labels.append(CLASS_TO_IDX[cls])
    return video_preds, video_labels


MODEL_NAMES = {
    1: "EfficientNet-B0",
    3: "TimeSformer",
    4: "Video Swin-T",
}


def _save_confusion_matrix(cm, class_names, out_path, title=""):
    cm_arr = np.array(cm, dtype=float)
    cm_norm = cm_arr / (cm_arr.sum(axis=1, keepdims=True) + 1e-6)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted", labelpad=10)
    ax.set_ylabel("True")
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── combined plots ────────────────────────────────────────────────────────────

def plot_combined_training_curves(model_ids: list):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors  = {1: "#1f77b4", 3: "#2ca02c", 4: "#d62728"}
    configs = {1: M1, 3: M3, 4: M4}

    for mid in model_ids:
        log_path = Path(configs[mid]["checkpoint_dir"]) / "training_log.csv"
        if not log_path.exists():
            continue
        epochs, tr_loss, val_loss, tr_acc, val_acc = [], [], [], [], []
        with open(log_path) as f:
            for row in csv.DictReader(f):
                epochs.append(int(row["epoch"]) + 1)
                tr_loss.append(float(row["train_loss"]))
                val_loss.append(float(row["val_loss"]))
                tr_acc.append(float(row["train_acc"]))
                val_acc.append(float(row["val_acc"]))

        c    = colors[mid]
        name = MODEL_NAMES[mid]
        axes[0].plot(epochs, tr_loss,  "-",  color=c, label=f"{name} train")
        axes[0].plot(epochs, val_loss, "--", color=c, label=f"{name} val", alpha=0.7)
        axes[1].plot(epochs, tr_acc,   "-",  color=c, label=f"{name} train")
        axes[1].plot(epochs, val_acc,  "--", color=c, label=f"{name} val", alpha=0.7)

    axes[0].set_title("Loss");     axes[0].set_xlabel("Epoch"); axes[0].legend(fontsize=8)
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend(fontsize=8)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    plt.suptitle("Training & Validation Curves")
    plt.tight_layout()
    out = PLOTS_DIR / "combined_training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_model_comparison_bar(results: dict):
    model_ids = sorted(results.keys())
    labels    = [MODEL_NAMES[m] for m in model_ids]
    x = np.arange(len(labels))

    metrics_to_plot = [
        ("accuracy",         "Accuracy",         "#1f77b4"),
        ("macro_f1",         "Macro-F1",         "#ff7f0e"),
        ("weighted_f1",      "Weighted-F1",      "#2ca02c"),
        ("balanced_accuracy","Balanced Acc",     "#9467bd"),
    ]
    n = len(metrics_to_plot)
    w = 0.8 / n

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 3), 5))
    for i, (key, display, color) in enumerate(metrics_to_plot):
        vals  = [results[m][key] for m in model_ids]
        offset = (i - n / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=display, color=color)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.4f}", ha="center", fontsize=7, rotation=90)

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))
    ax.set_title("Model Comparison — Test Set Metrics")
    ax.legend()
    plt.tight_layout()
    out = PLOTS_DIR / "model_comparison_bar.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_combined_confusion_matrices(results: dict):
    model_ids = sorted(results.keys())
    n = len(model_ids)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, mid in zip(axes, model_ids):
        cm = np.array(results[mid]["confusion_matrix"], dtype=float)
        cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-6)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
        ax.set_title(MODEL_NAMES[mid])
        ax.set_xlabel("Predicted", labelpad=8)
        ax.set_ylabel("True")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    plt.suptitle("Confusion Matrices — All Models", y=1.02)
    plt.tight_layout()
    out = PLOTS_DIR / "confusion_matrices.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_per_class_prf(results: dict):
    model_ids = sorted(results.keys())
    n = len(model_ids)
    metrics_def = [
        ("per_class_precision", "Precision", "#1f77b4"),
        ("per_class_recall",    "Recall",    "#ff7f0e"),
        ("per_class_f1",        "F1",        "#2ca02c"),
    ]
    x = np.arange(len(CLASSES))
    w = 0.8 / 3

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, mid in zip(axes, model_ids):
        for i, (key, label, color) in enumerate(metrics_def):
            vals = [results[mid][key].get(c, 0) for c in CLASSES]
            ax.bar(x + (i - 1) * w, vals, w, label=label, color=color, alpha=0.85)
        ax.set_title(MODEL_NAMES[mid])
        ax.set_xticks(x)
        ax.set_xticklabels(CLASSES, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        if ax is axes[0]:
            ax.legend(fontsize=9)

    plt.suptitle("Per-class Precision / Recall / F1")
    plt.tight_layout()
    out = PLOTS_DIR / "per_class_prf.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_tradeoff(results: dict):
    model_ids = sorted(results.keys())
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, mid in enumerate(model_ids):
        t = results[mid].get("inference_ms", 0)
        a = results[mid]["accuracy"]
        name = MODEL_NAMES[mid]
        ax.scatter(t, a, s=200, color=colors[i], zorder=5, label=name)
        ax.annotate(f"{name}\n{a:.1%}", (t, a),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.set_xlabel("Inference time (ms/sample)")
    ax.set_ylabel("Test Accuracy")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title("Speed vs Accuracy Trade-off")
    ax.legend()
    plt.tight_layout()
    out = PLOTS_DIR / "tradeoff.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=int, nargs="+", choices=[1, 3, 4],
                        required=True, help="Model(s) to evaluate: 1=CNN 3=TimeSformer 4=VideoSwin-T")
    args = parser.parse_args()

    splits  = get_splits()
    results = {}

    for mid in args.model:
        results[mid] = evaluate_model(mid, splits)

    if len(results) > 1:
        plot_combined_training_curves(args.model)
        plot_model_comparison_bar(results)
        plot_combined_confusion_matrices(results)
        plot_per_class_prf(results)
        plot_tradeoff(results)
        print(f"\nAll comparison plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
