from __future__ import annotations

import torch

from model_service.metrics import dice_score, iou_score


def test_metrics_return_one_for_perfect_overlap() -> None:
    prediction = torch.tensor([[[1.0, 0.0], [1.0, 1.0]]])
    target = torch.tensor([[[1.0, 0.0], [1.0, 1.0]]])

    assert dice_score(prediction, target) == 1.0
    assert iou_score(prediction, target) == 1.0
