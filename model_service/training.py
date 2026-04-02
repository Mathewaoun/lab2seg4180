from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn

from model_service.metrics import segmentation_metrics


class DiceBCELoss(nn.Module):
    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * targets).sum(dim=(1, 2, 3))
        denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - ((2.0 * intersection + self.eps) / (denominator + self.eps)).mean()
        return bce + dice_loss


def run_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_batches = 0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        if is_training:
            optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, masks)

        if is_training:
            loss.backward()
            optimizer.step()

        probabilities = torch.sigmoid(logits).detach().cpu()
        batch_masks = masks.detach().cpu()
        metrics = segmentation_metrics(probabilities, batch_masks, threshold=threshold)

        total_loss += loss.item()
        total_dice += metrics["dice_score"]
        total_iou += metrics["iou"]
        total_batches += 1

    if total_batches == 0:
        raise ValueError("The dataloader is empty.")

    return {
        "loss": total_loss / total_batches,
        "dice_score": total_dice / total_batches,
        "iou": total_iou / total_batches,
    }


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    threshold: float,
    checkpoint_path: str | Path,
) -> dict[str, list[float]]:
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "train_dice_score": [],
        "train_iou": [],
        "val_loss": [],
        "val_dice_score": [],
        "val_iou": [],
    }
    best_val_dice = -1.0

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            threshold=threshold,
            optimizer=optimizer,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                threshold=threshold,
            )

        history["train_loss"].append(train_metrics["loss"])
        history["train_dice_score"].append(train_metrics["dice_score"])
        history["train_iou"].append(train_metrics["iou"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_dice_score"].append(val_metrics["dice_score"])
        history["val_iou"].append(val_metrics["iou"])

        if val_metrics["dice_score"] > best_val_dice:
            best_val_dice = val_metrics["dice_score"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_dice_score": val_metrics["dice_score"],
                    "val_iou": val_metrics["iou"],
                    "threshold": threshold,
                },
                checkpoint_path,
            )

    return history


def save_training_artifacts(history: dict[str, list[float]], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "training_history.json"
    plot_path = output_dir / "training_curves.png"

    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train loss")
    axes[0].plot(epochs, history["val_loss"], label="Validation loss")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_dice_score"], label="Train Dice")
    axes[1].plot(epochs, history["val_dice_score"], label="Validation Dice")
    axes[1].plot(epochs, history["train_iou"], label="Train IoU")
    axes[1].plot(epochs, history["val_iou"], label="Validation IoU")
    axes[1].set_title("Segmentation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
