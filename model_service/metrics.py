from __future__ import annotations

import torch


def threshold_mask(mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (mask >= threshold).float()


def dice_score(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7) -> float:
    predictions = threshold_mask(predictions, threshold)
    targets = threshold_mask(targets, 0.5)
    intersection = (predictions * targets).sum().item()
    denominator = predictions.sum().item() + targets.sum().item()
    return float((2.0 * intersection + eps) / (denominator + eps))


def iou_score(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7) -> float:
    predictions = threshold_mask(predictions, threshold)
    targets = threshold_mask(targets, 0.5)
    intersection = (predictions * targets).sum().item()
    union = predictions.sum().item() + targets.sum().item() - intersection
    return float((intersection + eps) / (union + eps))


def segmentation_metrics(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    return {
        "dice_score": dice_score(predictions, targets, threshold=threshold),
        "iou": iou_score(predictions, targets, threshold=threshold),
    }
