from __future__ import annotations

import numpy as np

from scripts.prepare_dataset import bbox_to_mask, build_week7_mask_from_candidates, compute_iou


def test_bbox_to_mask_marks_expected_pixels() -> None:
    mask = bbox_to_mask([1, 2, 3, 2], (6, 6))

    assert mask.shape == (6, 6)
    assert mask.sum() == 6
    assert mask[2, 1]
    assert mask[3, 3]
    assert not mask[0, 0]


def test_compute_iou_returns_expected_overlap() -> None:
    mask_a = np.array([[1, 1], [0, 0]], dtype=bool)
    mask_b = np.array([[1, 0], [1, 0]], dtype=bool)

    assert compute_iou(mask_a, mask_b) == 1 / 3


def test_week7_mask_builder_prefers_sam_matches_and_falls_back_to_bbox() -> None:
    bbox_a = bbox_to_mask([0, 0, 2, 2], (5, 5))
    bbox_b = bbox_to_mask([3, 3, 2, 2], (5, 5))
    sam_match = bbox_to_mask([0, 0, 2, 2], (5, 5))
    sam_noise = bbox_to_mask([1, 3, 1, 1], (5, 5))

    final_mask, summary = build_week7_mask_from_candidates(
        bbox_masks=[bbox_a, bbox_b],
        sam_masks=[sam_match, sam_noise],
        iou_threshold=0.3,
    )

    assert final_mask.sum() == bbox_a.sum() + bbox_b.sum()
    assert summary["accepted_sam_masks"] == 1.0
    assert summary["matched_bboxes"] == 1.0
