from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_service.data import build_dataloader
from model_service.training import save_training_artifacts, train_model
from model_service.unet import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a U-Net house segmentation model.")
    parser.add_argument("--data-dir", default="data/processed", help="Prepared dataset root.")
    parser.add_argument("--output-dir", default="artifacts", help="Directory for checkpoints and plots.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--image-size", type=int, default=256, help="Training image size.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for metrics.")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or auto.")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    train_loader = build_dataloader(args.data_dir, "train", args.image_size, args.batch_size, shuffle=True)
    val_loader = build_dataloader(args.data_dir, "val", args.image_size, args.batch_size, shuffle=False)

    model = UNet().to(device)
    checkpoint_path = Path(args.output_dir) / "checkpoints" / "best_model.pt"
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        threshold=args.threshold,
        checkpoint_path=checkpoint_path,
    )
    save_training_artifacts(history, Path(args.output_dir) / "training")
    print(f"Training complete. Best checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
