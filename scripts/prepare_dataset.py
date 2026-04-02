from __future__ import annotations

import argparse
import json
import random
import time
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
HOUSE_LABELS = {"house", "building", "roof", "home"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare image/mask splits for house segmentation.")
    parser.add_argument(
        "--source",
        choices=("local", "week7_sam"),
        default="local",
        help="Use local image/annotation pairs or the Week 7 SAM pseudo-label workflow.",
    )
    parser.add_argument("--images-dir", help="Directory containing raw aerial images.")
    parser.add_argument(
        "--annotations-dir",
        help="Directory containing Week 7 masks or polygon annotations.",
    )
    parser.add_argument(
        "--hf-dataset-id",
        default="keremberke/satellite-building-segmentation",
        help="Hugging Face dataset to use for the Week 7 SAM workflow.",
    )
    parser.add_argument(
        "--hf-config",
        default="full",
        help="Dataset configuration name for the Week 7 SAM workflow.",
    )
    parser.add_argument(
        "--hf-splits",
        nargs="+",
        default=["train"],
        help="Dataset splits to load for the Week 7 SAM workflow.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit for quick experiments.",
    )
    parser.add_argument(
        "--sam-checkpoint",
        help="Path to the SAM checkpoint file, e.g. sam_vit_h_4b8939.pth.",
    )
    parser.add_argument(
        "--sam-model-type",
        default="vit_h",
        help="SAM model type such as vit_h, vit_l, or vit_b.",
    )
    parser.add_argument(
        "--sam-device",
        default="cpu",
        help="Device for SAM inference: cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--sam-points-per-side",
        type=int,
        default=32,
        help="SAM grid density. Lower values are faster but can reduce mask quality.",
    )
    parser.add_argument(
        "--sam-points-per-batch",
        type=int,
        default=64,
        help="Number of SAM prompt points processed together.",
    )
    parser.add_argument(
        "--sam-pred-iou-thresh",
        type=float,
        default=0.88,
        help="Minimum SAM-predicted IoU to keep a candidate mask.",
    )
    parser.add_argument(
        "--sam-stability-score-thresh",
        type=float,
        default=0.95,
        help="Minimum SAM stability score to keep a candidate mask.",
    )
    parser.add_argument(
        "--sam-crop-n-layers",
        type=int,
        default=0,
        help="Extra crop layers for SAM. Zero is faster and good for smoke tests.",
    )
    parser.add_argument(
        "--match-iou-threshold",
        type=float,
        default=0.30,
        help="Minimum IoU between a SAM mask and a label box mask to accept it.",
    )
    parser.add_argument(
        "--min-mask-area",
        type=int,
        default=0,
        help="Optional minimum area for accepted SAM masks.",
    )
    parser.add_argument(
        "--preserve-dataset-splits",
        action="store_true",
        help="Keep the original Hugging Face split names instead of re-splitting all samples.",
    )
    parser.add_argument("--output-dir", default="data/processed", help="Output directory for train/val/test splits.")
    parser.add_argument("--val-size", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test-size", type=float, default=0.15, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits.")
    parser.add_argument(
        "--target-rgb",
        type=int,
        nargs=3,
        default=None,
        metavar=("R", "G", "B"),
        help="Optional RGB color to treat as the house class in color masks.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress details while generating masks.",
    )
    return parser.parse_args()


def indexed_files(directory: Path, allowed_suffixes: set[str] | None = None) -> dict[str, Path]:
    allowed_suffixes = allowed_suffixes or IMAGE_EXTENSIONS.union({".json"})
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in allowed_suffixes
    }


def split_stems(stems: list[str], val_size: float, test_size: float, seed: int) -> dict[str, list[str]]:
    if val_size + test_size >= 1.0:
        raise ValueError("Validation and test split ratios must sum to less than 1.0.")

    shuffled = stems[:]
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    test_count = int(total * test_size)
    val_count = int(total * val_size)
    train_count = total - val_count - test_count
    if train_count <= 0:
        raise ValueError("Split ratios leave no data for training.")

    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def build_binary_mask(annotation_path: Path, image_size: tuple[int, int], target_rgb: list[int] | None) -> Image.Image:
    if annotation_path.suffix.lower() == ".json":
        return build_mask_from_json(annotation_path, image_size)
    return build_mask_from_image(annotation_path, target_rgb)


def build_mask_from_json(annotation_path: Path, image_size: tuple[int, int]) -> Image.Image:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)

    if "shapes" in payload:
        for shape in payload["shapes"]:
            label = str(shape.get("label", "")).lower()
            if label and label not in HOUSE_LABELS:
                continue
            points = shape.get("points", [])
            if not points:
                continue
            if shape.get("shape_type") == "rectangle" and len(points) == 2:
                (x1, y1), (x2, y2) = points
                points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            draw.polygon([tuple(point) for point in points], fill=255)
        return mask

    if "polygons" in payload:
        for polygon in payload["polygons"]:
            label = str(polygon.get("label", "house")).lower()
            if label not in HOUSE_LABELS:
                continue
            points = polygon.get("points") or polygon.get("vertices") or []
            if points:
                draw.polygon([tuple(point) for point in points], fill=255)
        return mask

    raise ValueError(
        f"Unsupported JSON annotation format in {annotation_path}. "
        "Update build_mask_from_json() to match your Week 7 export format."
    )


def build_mask_from_image(annotation_path: Path, target_rgb: list[int] | None) -> Image.Image:
    mask_image = Image.open(annotation_path)
    if target_rgb is not None:
        rgb_array = np.asarray(mask_image.convert("RGB"), dtype=np.uint8)
        binary_mask = np.all(rgb_array == np.array(target_rgb, dtype=np.uint8), axis=-1)
    else:
        grayscale = np.asarray(mask_image.convert("L"), dtype=np.uint8)
        binary_mask = grayscale > 0
    return Image.fromarray((binary_mask.astype(np.uint8) * 255), mode="L")


def bbox_to_mask(bbox: list[float] | tuple[float, float, float, float], image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    x_min, y_min, box_width, box_height = [int(round(value)) for value in bbox]
    x_max = max(x_min, min(width, x_min + box_width))
    y_max = max(y_min, min(height, y_min + box_height))
    x_min = max(0, min(width, x_min))
    y_min = max(0, min(height, y_min))
    mask = np.zeros((height, width), dtype=bool)
    mask[y_min:y_max, x_min:x_max] = True
    return mask


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    intersection = np.logical_and(mask_a, mask_b).sum()
    return float(intersection / union)


def build_week7_mask_from_candidates(
    bbox_masks: list[np.ndarray],
    sam_masks: list[np.ndarray],
    iou_threshold: float,
    min_mask_area: int = 0,
) -> tuple[np.ndarray, dict[str, float]]:
    final_mask = np.zeros_like(bbox_masks[0], dtype=bool) if bbox_masks else np.zeros((0, 0), dtype=bool)
    matched_bbox_indices: set[int] = set()
    accepted_masks = 0
    best_ious: list[float] = []

    for sam_mask in sam_masks:
        sam_mask = sam_mask.astype(bool)
        if sam_mask.sum() <= min_mask_area:
            continue

        best_bbox_idx = -1
        best_iou = 0.0
        for bbox_index, bbox_mask in enumerate(bbox_masks):
            iou = compute_iou(sam_mask, bbox_mask)
            if iou > best_iou:
                best_iou = iou
                best_bbox_idx = bbox_index

        if best_iou >= iou_threshold and best_bbox_idx >= 0:
            final_mask |= sam_mask
            matched_bbox_indices.add(best_bbox_idx)
            accepted_masks += 1
            best_ious.append(best_iou)

    for bbox_index, bbox_mask in enumerate(bbox_masks):
        if bbox_index not in matched_bbox_indices:
            final_mask |= bbox_mask

    summary = {
        "accepted_sam_masks": float(accepted_masks),
        "matched_bboxes": float(len(matched_bbox_indices)),
        "avg_best_iou": float(sum(best_ious) / len(best_ious)) if best_ious else 0.0,
    }
    return final_mask, summary


def _load_week7_dependencies() -> tuple[Any, Any, Any]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "The Week 7 SAM workflow requires the 'huggingface_hub' package. "
            "Install it with: pip install -r requirements-sam.txt"
        ) from exc

    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError as exc:
        raise RuntimeError(
            "The Week 7 SAM workflow requires the Segment Anything package. "
            "Install it with: pip install -r requirements-sam.txt"
        ) from exc

    return hf_hub_download, SamAutomaticMaskGenerator, sam_model_registry


def coerce_sam_mask_generator_precision(mask_generator: Any, device_name: str) -> Any:
    if device_name != "mps":
        return mask_generator

    if hasattr(mask_generator, "point_grids"):
        mask_generator.point_grids = [
            np.asarray(grid, dtype=np.float32) for grid in mask_generator.point_grids
        ]

    transform = getattr(getattr(mask_generator, "predictor", None), "transform", None)
    if transform is None or not hasattr(transform, "apply_coords"):
        return mask_generator

    original_apply_coords = transform.apply_coords

    def apply_coords_float32(coords: np.ndarray, original_size: tuple[int, ...]) -> np.ndarray:
        transformed = original_apply_coords(coords, original_size)
        return np.asarray(transformed, dtype=np.float32)

    transform.apply_coords = apply_coords_float32
    return mask_generator


def collect_hf_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    hf_hub_download, _, _ = _load_week7_dependencies()
    records: list[dict[str, Any]] = []
    for split_name in args.hf_splits:
        archive_name = f"data/{split_name}.zip"
        zip_path = hf_hub_download(
            repo_id=args.hf_dataset_id,
            repo_type="dataset",
            filename=archive_name,
        )
        with zipfile.ZipFile(zip_path) as zf:
            coco_payload = json.loads(zf.read("_annotations.coco.json"))
            annotations_by_image_id: dict[int, list[list[float]]] = defaultdict(list)
            for annotation in coco_payload["annotations"]:
                annotations_by_image_id[annotation["image_id"]].append(annotation["bbox"])

            for index, image_info in enumerate(coco_payload["images"]):
                with zf.open(image_info["file_name"]) as image_file:
                    image = Image.open(BytesIO(image_file.read())).convert("RGB")

                source_split = split_name
                if split_name == "valid":
                    source_split = "val"
                if split_name == "valid-mini":
                    source_split = "val"

                stem = Path(image_info["file_name"]).stem
                if source_split != split_name:
                    stem = f"{split_name}_{stem}"

                bboxes = annotations_by_image_id.get(image_info["id"], [])
                if args.limit is not None and len(records) >= args.limit:
                    return records

                records.append(
                    {
                        "stem": stem,
                        "source_split": source_split,
                        "image": image,
                        "bboxes": bboxes,
                    }
                )
                if args.limit is not None and len(records) >= args.limit:
                    return records
    return records


def build_week7_sam_mask(
    image: Image.Image,
    bboxes: list[list[float]],
    mask_generator: Any,
    iou_threshold: float,
    min_mask_area: int,
) -> tuple[Image.Image, dict[str, float]]:
    image_np = np.asarray(image.convert("RGB"))
    sam_predictions = mask_generator.generate(image_np)
    bbox_masks = [bbox_to_mask(bbox, image.size) for bbox in bboxes]
    if not bbox_masks:
        empty_mask = np.zeros((image.height, image.width), dtype=np.uint8)
        return Image.fromarray(empty_mask * 255, mode="L"), {
            "accepted_sam_masks": 0.0,
            "matched_bboxes": 0.0,
            "avg_best_iou": 0.0,
            "num_bboxes": 0.0,
            "num_sam_candidates": float(len(sam_predictions)),
        }
    sam_masks = [prediction["segmentation"] for prediction in sam_predictions]
    final_mask, summary = build_week7_mask_from_candidates(
        bbox_masks=bbox_masks,
        sam_masks=sam_masks,
        iou_threshold=iou_threshold,
        min_mask_area=min_mask_area,
    )
    summary["num_bboxes"] = float(len(bboxes))
    summary["num_sam_candidates"] = float(len(sam_predictions))
    return Image.fromarray((final_mask.astype(np.uint8) * 255), mode="L"), summary


def write_dataset_split(
    output_dir: Path,
    split: str,
    image_name: str,
    image: Image.Image,
    mask: Image.Image,
) -> None:
    images_dir = output_dir / split / "images"
    masks_dir = output_dir / split / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    image.save(images_dir / image_name)
    mask.save(masks_dir / f"{Path(image_name).stem}.png")


def prepare_local_dataset(args: argparse.Namespace) -> dict[str, Any]:
    if not args.images_dir or not args.annotations_dir:
        raise ValueError("--images-dir and --annotations-dir are required when --source=local.")

    images_dir = Path(args.images_dir)
    annotations_dir = Path(args.annotations_dir)
    output_dir = Path(args.output_dir)

    image_map = indexed_files(images_dir, IMAGE_EXTENSIONS)
    annotation_map = indexed_files(annotations_dir)
    common_stems = sorted(set(image_map) & set(annotation_map))
    if not common_stems:
        raise ValueError("No matching image/annotation pairs were found.")

    split_map = split_stems(common_stems, args.val_size, args.test_size, args.seed)

    for split, stems in split_map.items():
        for stem in stems:
            image_path = image_map[stem]
            annotation_path = annotation_map[stem]
            image = Image.open(image_path).convert("RGB")
            mask = build_binary_mask(annotation_path, image.size, args.target_rgb)
            write_dataset_split(output_dir, split, image_path.name, image, mask)

    return {
        "source": "local",
        "total_samples": len(common_stems),
        "train_samples": len(split_map["train"]),
        "val_samples": len(split_map["val"]),
        "test_samples": len(split_map["test"]),
        "output_dir": str(output_dir),
    }


def prepare_week7_sam_dataset(args: argparse.Namespace) -> dict[str, Any]:
    if not args.sam_checkpoint:
        raise ValueError("--sam-checkpoint is required when --source=week7_sam.")

    _, SamAutomaticMaskGenerator, sam_model_registry = _load_week7_dependencies()

    output_dir = Path(args.output_dir)
    checkpoint_path = Path(args.sam_checkpoint)
    if not checkpoint_path.exists():
        raise ValueError(f"SAM checkpoint not found at {checkpoint_path}")

    sam = sam_model_registry[args.sam_model_type](checkpoint=str(checkpoint_path))
    sam.to(device=args.sam_device)
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=args.sam_points_per_side,
        points_per_batch=args.sam_points_per_batch,
        pred_iou_thresh=args.sam_pred_iou_thresh,
        stability_score_thresh=args.sam_stability_score_thresh,
        crop_n_layers=args.sam_crop_n_layers,
    )
    mask_generator = coerce_sam_mask_generator_precision(mask_generator, args.sam_device)

    records = collect_hf_records(args)
    if not records:
        raise ValueError("No dataset samples were loaded from Hugging Face.")
    if args.verbose:
        print(
            f"Loaded {len(records)} records from {args.hf_dataset_id} using splits {args.hf_splits}.",
            flush=True,
        )

    if args.preserve_dataset_splits:
        split_assignments = {
            "train": [record for record in records if record["source_split"] == "train"],
            "val": [record for record in records if record["source_split"] in {"validation", "val"}],
            "test": [record for record in records if record["source_split"] == "test"],
        }
        if not split_assignments["val"]:
            split_assignments["val"] = []
        if not split_assignments["test"]:
            split_assignments["test"] = []
    else:
        stems = [record["stem"] for record in records]
        split_map = split_stems(stems, args.val_size, args.test_size, args.seed)
        record_lookup = {record["stem"]: record for record in records}
        split_assignments = {
            split: [record_lookup[stem] for stem in stems_for_split]
            for split, stems_for_split in split_map.items()
        }

    all_summaries: list[dict[str, Any]] = []
    total_records = sum(len(split_records) for split_records in split_assignments.values())
    processed_count = 0
    for split, split_records in split_assignments.items():
        for record in split_records:
            processed_count += 1
            image = record["image"]
            start_time = time.perf_counter()
            if args.verbose:
                print(
                    f"[{processed_count}/{total_records}] Processing {record['stem']} -> split={split} "
                    f"size={image.size} bboxes={len(record['bboxes'])}",
                    flush=True,
                )
            mask, mask_summary = build_week7_sam_mask(
                image=image,
                bboxes=record["bboxes"],
                mask_generator=mask_generator,
                iou_threshold=args.match_iou_threshold,
                min_mask_area=args.min_mask_area,
            )
            image_name = f"{record['stem']}.png"
            write_dataset_split(output_dir, split, image_name, image, mask)
            duration = time.perf_counter() - start_time
            if args.verbose:
                print(
                    f"Saved {image_name} in {duration:.1f}s "
                    f"(sam_candidates={int(mask_summary['num_sam_candidates'])}, "
                    f"accepted={int(mask_summary['accepted_sam_masks'])}, "
                    f"avg_best_iou={mask_summary['avg_best_iou']:.3f})",
                    flush=True,
                )
            all_summaries.append(
                {
                    "stem": record["stem"],
                    "split": split,
                    **mask_summary,
                }
            )

    avg_iou = sum(item["avg_best_iou"] for item in all_summaries) / len(all_summaries)
    total_masks = sum(item["accepted_sam_masks"] for item in all_summaries)
    summary = {
        "source": "week7_sam",
        "dataset_id": args.hf_dataset_id,
        "dataset_config": args.hf_config,
        "sam_checkpoint": str(checkpoint_path),
        "sam_model_type": args.sam_model_type,
        "sam_device": args.sam_device,
        "sam_points_per_side": args.sam_points_per_side,
        "sam_points_per_batch": args.sam_points_per_batch,
        "sam_pred_iou_thresh": args.sam_pred_iou_thresh,
        "sam_stability_score_thresh": args.sam_stability_score_thresh,
        "sam_crop_n_layers": args.sam_crop_n_layers,
        "match_iou_threshold": args.match_iou_threshold,
        "total_samples": len(all_summaries),
        "train_samples": len(split_assignments["train"]),
        "val_samples": len(split_assignments["val"]),
        "test_samples": len(split_assignments["test"]),
        "avg_best_iou": avg_iou,
        "avg_accepted_sam_masks_per_image": total_masks / len(all_summaries),
        "output_dir": str(output_dir),
    }
    (output_dir / "week7_sam_mask_summary.json").write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "week7_sam":
        summary = prepare_week7_sam_dataset(args)
    else:
        summary = prepare_local_dataset(args)

    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
