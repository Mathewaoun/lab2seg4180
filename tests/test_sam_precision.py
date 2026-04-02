from __future__ import annotations

import numpy as np

from scripts.prepare_dataset import coerce_sam_mask_generator_precision


class DummyTransform:
    def apply_coords(self, coords: np.ndarray, original_size: tuple[int, ...]) -> np.ndarray:
        return np.asarray(coords, dtype=np.float64) * 2.0


class DummyPredictor:
    def __init__(self) -> None:
        self.transform = DummyTransform()


class DummyMaskGenerator:
    def __init__(self) -> None:
        self.point_grids = [np.array([[0.1, 0.2]], dtype=np.float64)]
        self.predictor = DummyPredictor()


def test_coerce_sam_mask_generator_precision_updates_point_grids_and_coords() -> None:
    generator = DummyMaskGenerator()

    generator = coerce_sam_mask_generator_precision(generator, "mps")
    transformed = generator.predictor.transform.apply_coords(np.array([[1.0, 2.0]]), (500, 500))

    assert generator.point_grids[0].dtype == np.float32
    assert transformed.dtype == np.float32


def test_coerce_sam_mask_generator_precision_skips_non_mps() -> None:
    generator = DummyMaskGenerator()

    generator = coerce_sam_mask_generator_precision(generator, "cpu")
    transformed = generator.predictor.transform.apply_coords(np.array([[1.0, 2.0]]), (500, 500))

    assert generator.point_grids[0].dtype == np.float64
    assert transformed.dtype == np.float64
