"""Reproducible 2D classification training loop and checkpoint management for NeuroVR."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.classifier_2d import CLASS_NAMES, CLASS_TO_INDEX


class Trainer2D:
    """Encapsulates training, validation, checkpointing, and metric logging."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        config: dict[str, Any],
        scheduler: Any = None,
        checkpoint_dir: str | Path = "checkpoints/classification",
        report_dir: str | Path = "reports/classification",
        model_name: str = "EfficientNet-B4",
        image_size: tuple[int, int] = (380, 380),
        monitor_metric: str = "val_loss",
        monitor_mode: str = "min",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.scheduler = scheduler
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.image_size = tuple(image_size)
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode

        self.history: list[dict[str, Any]] = []
        self.best_metric_value: float = float("inf") if monitor_mode == "min" else float("-inf")
        self.best_epoch: int = 0

    def train_epoch(self, max_batches: int | None = None) -> tuple[float, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total_samples += batch_size

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = correct / total_samples if total_samples > 0 else 0.0
        return avg_loss, accuracy

    def validate_epoch(self, max_batches: int | None = None) -> tuple[float, float]:
        """Run one validation epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(self.val_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total_samples += batch_size

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = correct / total_samples if total_samples > 0 else 0.0
        return avg_loss, accuracy

    def save_checkpoint(
        self,
        filepath: Path,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ) -> None:
        """Serialize structured checkpoint with complete metadata."""
        state = {
            "model_architecture": self.model_name,
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "class_names": list(CLASS_NAMES),
            "class_to_index": dict(CLASS_TO_INDEX),
            "image_size": list(self.image_size),
            "training_config": {
                "optimizer": self.optimizer.__class__.__name__,
                "lr": self.optimizer.param_groups[0]["lr"],
                "weight_decay": self.optimizer.param_groups[0].get("weight_decay", 0.0),
                "device": str(self.device),
            },
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "dataset_info": {
                "train_samples": len(self.train_loader.dataset) if hasattr(self.train_loader, "dataset") else 0,
                "val_samples": len(self.val_loader.dataset) if hasattr(self.val_loader, "dataset") else 0,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        torch.save(state, filepath)

    def train(
        self,
        epochs: int = 1,
        max_batches_per_epoch: int | None = None,
        verbose: bool = True,
    ) -> list[dict[str, Any]]:
        """Run the full training loop over configured epochs."""
        for epoch in range(1, epochs + 1):
            current_lr = self.optimizer.param_groups[0]["lr"]
            train_loss, train_acc = self.train_epoch(max_batches=max_batches_per_epoch)
            val_loss, val_acc = self.validate_epoch(max_batches=max_batches_per_epoch)

            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_accuracy": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_accuracy": round(val_acc, 6),
                "learning_rate": current_lr,
            }
            self.history.append(epoch_record)

            train_metrics = {"loss": train_loss, "accuracy": train_acc}
            val_metrics = {"loss": val_loss, "accuracy": val_acc}

            # Save last checkpoint
            last_path = self.checkpoint_dir / "last.pt"
            self.save_checkpoint(last_path, epoch, train_metrics, val_metrics)

            # Check if this epoch is the best
            current_metric = val_loss if self.monitor_metric == "val_loss" else val_acc
            is_better = (
                current_metric < self.best_metric_value
                if self.monitor_mode == "min"
                else current_metric > self.best_metric_value
            )
            if is_better or epoch == 1:
                self.best_metric_value = current_metric
                self.best_epoch = epoch
                best_path = self.checkpoint_dir / "best.pt"
                self.save_checkpoint(best_path, epoch, train_metrics, val_metrics)

            if verbose:
                print(
                    f"Epoch {epoch:02d}/{epochs:02d} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
                    f"LR: {current_lr:.6f}"
                )

        # Save persistent training history
        self.save_history()
        self.plot_curves()
        return self.history

    def save_history(self) -> Path:
        """Write training history JSON to report directory."""
        history_path = self.report_dir / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump({
                "model_name": self.model_name,
                "best_epoch": self.best_epoch,
                "best_metric": self.monitor_metric,
                "best_metric_value": self.best_metric_value,
                "history": self.history,
            }, f, indent=2)
        return history_path

    def plot_curves(self) -> Path | None:
        """Generate training and validation loss/accuracy curves."""
        if not self.history:
            return None
        epochs = [r["epoch"] for r in self.history]
        train_loss = [r["train_loss"] for r in self.history]
        val_loss = [r["val_loss"] for r in self.history]
        train_acc = [r["train_accuracy"] for r in self.history]
        val_acc = [r["val_accuracy"] for r in self.history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(epochs, train_loss, label="Train Loss", marker="o", color="#e74c3c")
        ax1.plot(epochs, val_loss, label="Val Loss", marker="o", color="#3498db")
        ax1.set_title(f"{self.model_name} — Loss Curves")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()

        ax2.plot(epochs, train_acc, label="Train Accuracy", marker="o", color="#2ecc71")
        ax2.plot(epochs, val_acc, label="Val Accuracy", marker="o", color="#f39c12")
        ax2.set_title(f"{self.model_name} — Accuracy Curves")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        plot_path = self.report_dir / "training_curves.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        return plot_path
