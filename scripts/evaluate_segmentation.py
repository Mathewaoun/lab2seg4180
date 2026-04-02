from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_service.data import SegmentationDataset
from model_service.metrics import segmentation_metrics
from model_service.unet import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained U-Net on the test split.")
    parser.add_argument("--data-dir", default="data/processed", help="Prepared dataset root.")
    parser.add_argument("--weights", default="artifacts/checkpoints/best_model.pt", help="Checkpoint path.")
    parser.add_argument("--output-dir", default="artifacts/evaluation", help="Evaluation output directory.")
    parser.add_argument("--image-size", type=int, default=256, help="Evaluation image size.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold.")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or auto.")
    parser.add_argument("--num-visualizations", type=int, default=3, help="How many prediction panels to save.")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def load_model(weights_path: str | Path, device: torch.device) -> UNet:
    model = UNet().to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def save_prediction_panel(image: torch.Tensor, target: torch.Tensor, prediction: torch.Tensor, output_path: Path) -> None:
    rgb = image.permute(1, 2, 0).numpy()
    ground_truth = target.squeeze(0).numpy()
    predicted = prediction.squeeze(0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("Image")
    axes[1].imshow(ground_truth, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[2].imshow(predicted, cmap="gray")
    axes[2].set_title("Prediction")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    dataset = SegmentationDataset(
        images_dir=Path(args.data_dir) / "test" / "images",
        masks_dir=Path(args.data_dir) / "test" / "masks",
        image_size=args.image_size,
    )
    model = load_model(args.weights, device)

    total_dice = 0.0
    total_iou = 0.0

    with torch.no_grad():
        for index in range(len(dataset)):
            image, target = dataset[index]
            logits = model(image.unsqueeze(0).to(device))
            probabilities = torch.sigmoid(logits).cpu().squeeze(0)
            prediction = (probabilities >= args.threshold).float()
            metrics = segmentation_metrics(prediction, target, threshold=args.threshold)
            total_dice += metrics["dice_score"]
            total_iou += metrics["iou"]

            if index < args.num_visualizations:
                save_prediction_panel(
                    image=image,
                    target=target,
                    prediction=prediction,
                    output_path=output_dir / f"prediction_{index + 1}.png",
                )

    results = {
        "num_test_samples": len(dataset),
        "dice_score": total_dice / len(dataset),
        "iou": total_iou / len(dataset),
        "weights": str(args.weights),
    }
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
