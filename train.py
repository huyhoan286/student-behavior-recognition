import argparse
import sys

import torch
from torch.utils.data import DataLoader

from config import M1, M3, M4
from data.splits import get_splits
from training.trainer import Trainer


def build_and_train(model_id: int, splits: dict, force: bool):
    from data.frame_extractor import extract_all_frames
    extract_all_frames(force=force)

    if model_id == 1:
        from data.dataset_m1 import FrameDataset
        from models.model1 import Model1

        train_ds = FrameDataset(splits["train"], split="train")
        val_ds   = FrameDataset(splits["val"],   split="val")
        train_loader = DataLoader(train_ds, batch_size=M1["batch_size"],
                                  shuffle=True,  num_workers=8, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=M1["batch_size"] * 2,
                                  shuffle=False, num_workers=4, pin_memory=True)
        model     = Model1()
        optimizer = torch.optim.AdamW(model.parameters(), lr=M1["lr"],
                                      weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=M1["epochs"])
        cfg = M1

    elif model_id == 3:
        from data.dataset_video import VideoClipDataset
        from models.model3 import Model3

        train_ds = VideoClipDataset(splits["train"], "train",
                                    M3["num_frames"], M3["mean"], M3["std"])
        val_ds   = VideoClipDataset(splits["val"],   "val",
                                    M3["num_frames"], M3["mean"], M3["std"])
        train_loader = DataLoader(train_ds, batch_size=M3["batch_size"],
                                  shuffle=True,  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=M3["batch_size"],
                                  shuffle=False, num_workers=2, pin_memory=True)
        model     = Model3()
        optimizer = torch.optim.AdamW(model.parameters(), lr=M3["lr"],
                                      weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=M3["epochs"])
        cfg = M3

    elif model_id == 4:
        from data.dataset_video import VideoClipDataset
        from models.model4 import Model4

        train_ds = VideoClipDataset(splits["train"], "train",
                                    M4["num_frames"], M4["mean"], M4["std"])
        val_ds   = VideoClipDataset(splits["val"],   "val",
                                    M4["num_frames"], M4["mean"], M4["std"])
        train_loader = DataLoader(train_ds, batch_size=M4["batch_size"],
                                  shuffle=True,  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=M4["batch_size"],
                                  shuffle=False, num_workers=2, pin_memory=True)
        model = Model4()
        # Differential LR: encoder nhỏ hơn, head lớn hơn
        optimizer = torch.optim.AdamW([
            {"params": model.get_encoder_params(), "lr": M4["backbone_lr"]},
            {"params": model.get_head_params(),    "lr": M4["lr"]},
        ], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=M4["epochs"])
        cfg = M4

    else:
        sys.exit(f"Invalid model id: {model_id}. Choose from 1, 3, 4.")

    trainer = Trainer(model, optimizer, scheduler, train_loader, val_loader, cfg)
    trainer.fit()


def main():
    parser = argparse.ArgumentParser(description="Train behavior recognition model")
    parser.add_argument("--model", type=int, choices=[1, 3, 4], required=True,
                        help="1=CNN  3=TimeSformer  4=VideoSwin-T")
    parser.add_argument("--force", action="store_true",
                        help="Ignore cache, recompute everything from scratch")
    args = parser.parse_args()

    splits = get_splits(force=args.force)
    build_and_train(args.model, splits, force=args.force)


if __name__ == "__main__":
    main()
