from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model_service.config import Settings
from model_service.data import pil_image_to_tensor, pil_mask_to_tensor
from model_service.metrics import segmentation_metrics
from model_service.unet import UNet


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _encode_mask(mask: np.ndarray) -> str:
    buffer = BytesIO()
    Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class SegmentationPredictor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = _resolve_device(settings.model_device)
        self.model = UNet()
        self.weights_path = Path(settings.model_weights_path)
        self.is_ready = False
        self.status_message = f"Checkpoint not found at {self.weights_path}."
        if self.weights_path.exists():
            checkpoint = torch.load(self.weights_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            self.is_ready = True
            self.status_message = f"Loaded weights from {self.weights_path}."

    def predict(self, image_bytes: bytes, ground_truth_bytes: bytes | None = None) -> dict[str, object]:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        original_width, original_height = image.size
        image_tensor = pil_image_to_tensor(image, self.settings.prediction_size).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.sigmoid(logits).cpu()

        prediction = (probabilities >= self.settings.prediction_threshold).float()
        prediction_array = prediction.squeeze(0).squeeze(0).numpy().astype(np.uint8)
        resized_prediction = np.asarray(
            Image.fromarray(prediction_array * 255, mode="L").resize(
                (original_width, original_height),
                Image.NEAREST,
            ),
            dtype=np.uint8,
        )
        resized_prediction = (resized_prediction > 0).astype(np.uint8)

        response: dict[str, object] = {
            "prediction_size": self.settings.prediction_size,
            "foreground_ratio": float(resized_prediction.mean()),
            "mask_png_base64": _encode_mask(resized_prediction),
            "message": "Segmentation generated successfully.",
        }

        if ground_truth_bytes is not None:
            ground_truth_image = Image.open(BytesIO(ground_truth_bytes))
            ground_truth_tensor = pil_mask_to_tensor(ground_truth_image, self.settings.prediction_size)
            response.update(segmentation_metrics(prediction.squeeze(0), ground_truth_tensor, self.settings.prediction_threshold))

        return response
