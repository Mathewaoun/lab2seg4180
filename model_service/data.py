from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _indexed_files(directory: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def paired_image_mask_paths(images_dir: str | Path, masks_dir: str | Path) -> list[tuple[Path, Path]]:
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    image_map = _indexed_files(images_dir)
    mask_map = _indexed_files(masks_dir)
    common_stems = sorted(set(image_map) & set(mask_map))
    return [(image_map[stem], mask_map[stem]) for stem in common_stems]


def pil_image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1))


def pil_mask_to_tensor(mask: Image.Image, image_size: int) -> torch.Tensor:
    mask = mask.convert("L").resize((image_size, image_size), Image.NEAREST)
    array = (np.asarray(mask, dtype=np.float32) > 0).astype(np.float32)
    return torch.from_numpy(array).unsqueeze(0)


class SegmentationDataset(Dataset):
    def __init__(self, images_dir: str | Path, masks_dir: str | Path, image_size: int = 256) -> None:
        self.pairs = paired_image_mask_paths(images_dir, masks_dir)
        if not self.pairs:
            raise ValueError(f"No image/mask pairs found in {images_dir} and {masks_dir}.")
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[index]
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        return pil_image_to_tensor(image, self.image_size), pil_mask_to_tensor(mask, self.image_size)


def build_dataloader(
    data_dir: str | Path,
    split: str,
    image_size: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    data_dir = Path(data_dir)
    dataset = SegmentationDataset(
        images_dir=data_dir / split / "images",
        masks_dir=data_dir / split / "masks",
        image_size=image_size,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
