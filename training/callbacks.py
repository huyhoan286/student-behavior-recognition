import torch
from pathlib import Path


class ModelCheckpoint:
    def __init__(self, ckpt_dir: Path, model, optimizer, mode: str = "max"):
        self.ckpt_dir  = ckpt_dir
        self.model     = model
        self.optimizer = optimizer
        self.mode      = mode
        self.best_val  = float("-inf") if mode == "max" else float("inf")
        self.start_epoch = 0
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._try_resume()

    def _try_resume(self):
        last_path = self.ckpt_dir / "last.pth"
        if last_path.exists():
            ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.best_val    = ckpt.get("best_val", self.best_val)
            self.start_epoch = ckpt["epoch"] + 1
            print(f"[ckpt] Resumed from epoch {ckpt['epoch']} (best_val={self.best_val:.4f})")

    def step(self, epoch: int, val_metric: float) -> bool:
        improved = (self.mode == "max" and val_metric > self.best_val) or \
                   (self.mode == "min" and val_metric < self.best_val)
        if improved:
            self.best_val = val_metric
            torch.save(self._state(epoch), self.ckpt_dir / "best.pth")

        torch.save(self._state(epoch), self.ckpt_dir / "last.pth")
        return improved

    def _state(self, epoch: int) -> dict:
        m = self.model
        if hasattr(m, "module"):   # DataParallel
            m = m.module
        return {
            "epoch":     epoch,
            "model":     m.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_val":  self.best_val,
        }


class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "max"):
        self.patience = patience
        self.mode     = mode
        self.best     = float("-inf") if mode == "max" else float("inf")
        self.counter  = 0

    def step(self, val_metric: float) -> bool:
        improved = (self.mode == "max" and val_metric > self.best) or \
                   (self.mode == "min" and val_metric < self.best)
        if improved:
            self.best    = val_metric
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
