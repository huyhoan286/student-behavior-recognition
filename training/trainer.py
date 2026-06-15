import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from config import IDX_TO_CLASS, NUM_CLASSES
from training.callbacks import EarlyStopping, ModelCheckpoint
from training.metrics import compute_metrics

console = Console()


class Trainer:
    def __init__(self, model, optimizer, scheduler,
                 train_loader, val_loader, config: dict):
        self.config      = config
        self.device      = "cuda" if torch.cuda.is_available() else "cpu"
        self.loss_fn     = nn.CrossEntropyLoss()
        self.scaler      = GradScaler("cuda")

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        self.model = model.to(self.device)

        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.train_loader = train_loader
        self.val_loader   = val_loader

        ckpt_dir = Path(config["checkpoint_dir"])
        self.checkpoint   = ModelCheckpoint(ckpt_dir, self.model, self.optimizer)
        self.early_stop   = EarlyStopping(patience=10)

        self.log_path = ckpt_dir / "training_log.csv"
        self._init_log()

    def _init_log(self):
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "train_acc",
                                  "val_loss", "val_acc", "val_f1", "time_s"])

    def _log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, val_f1, elapsed):
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{train_loss:.4f}", f"{train_acc:.4f}",
                f"{val_loss:.4f}", f"{val_acc:.4f}", f"{val_f1:.4f}", f"{elapsed:.1f}",
            ])

    def _print_epoch(self, epoch, total_epochs, train_loss, train_acc,
                     val_loss, val_acc, val_f1, improved):
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Epoch", style="bold")
        table.add_column("Train Loss")
        table.add_column("Train Acc")
        table.add_column("Val Loss")
        table.add_column("Val Acc")
        table.add_column("Val F1")
        table.add_column("")

        star = "★" if improved else ""
        table.add_row(
            f"{epoch+1}/{total_epochs}",
            f"{train_loss:.4f}", f"{train_acc:.2%}",
            f"{val_loss:.4f}",  f"{val_acc:.2%}",
            f"{val_f1:.4f}", star,
        )
        console.print(table)

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} train",
                    leave=False, unit="batch")
        for batch in pbar:
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            self.optimizer.zero_grad()
            with autocast("cuda"):
                logits = self.model(x)
                loss   = self.loss_fn(logits, y)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            bs = y.size(0)
            total_loss += loss.item() * bs
            correct    += (logits.argmax(1) == y).sum().item()
            total      += bs
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct/total:.2%}")

        return total_loss / total, correct / total

    @torch.no_grad()
    def _val_epoch(self):
        self.model.eval()
        total_loss, all_preds, all_labels = 0.0, [], []
        for batch in self.val_loader:
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            with autocast("cuda"):
                logits = self.model(x)
                loss   = self.loss_fn(logits, y)
            total_loss  += loss.item() * y.size(0)
            all_preds   += logits.argmax(1).cpu().tolist()
            all_labels  += y.cpu().tolist()

        metrics = compute_metrics(all_preds, all_labels,
                                  list(IDX_TO_CLASS.values()))
        avg_loss = total_loss / len(all_labels)
        return avg_loss, metrics["accuracy"], metrics["macro_f1"]

    def fit(self):
        total_epochs = self.config["epochs"]
        start_epoch  = self.checkpoint.start_epoch

        for epoch in range(start_epoch, total_epochs):
            t0 = time.time()

            # freeze / unfreeze backbone for model1/model2
            freeze_epochs = self.config.get("freeze_epochs", 0)
            if freeze_epochs > 0:
                backbone = getattr(self.model, "module", self.model).backbone
                requires_grad = epoch >= freeze_epochs
                for p in backbone.parameters():
                    p.requires_grad = requires_grad

            train_loss, train_acc = self._train_epoch(epoch)
            val_loss, val_acc, val_f1 = self._val_epoch()

            if self.scheduler:
                self.scheduler.step()

            improved = self.checkpoint.step(epoch, val_acc)
            self._log_epoch(epoch, train_loss, train_acc,
                            val_loss, val_acc, val_f1, time.time() - t0)
            self._print_epoch(epoch, total_epochs, train_loss, train_acc,
                              val_loss, val_acc, val_f1, improved)

            if self.early_stop.step(val_acc):
                console.print(f"[yellow]Early stopping at epoch {epoch+1}[/]")
                break

        console.print(f"[green]Best val acc: {self.checkpoint.best_val:.2%}[/]")
