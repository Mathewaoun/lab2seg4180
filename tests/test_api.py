from __future__ import annotations

import io

from PIL import Image

from app import create_app
from model_service.config import Settings


class DummyPredictor:
    def __init__(self, is_ready: bool = True) -> None:
        self.is_ready = is_ready
        self.status_message = "dummy predictor"

    def predict(self, image_bytes: bytes, ground_truth_bytes: bytes | None = None) -> dict[str, object]:
        return {
            "message": "Segmentation generated successfully.",
            "foreground_ratio": 0.25,
            "mask_png_base64": "ZmFrZS1tYXNr",
        }


def make_test_image() -> io.BytesIO:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), color=(255, 255, 255)).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def test_health_endpoint_reports_predictor_status() -> None:
    app = create_app(Settings(), predictor=DummyPredictor(is_ready=True))
    client = app.test_client()

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["model_loaded"] is True
    assert payload["status"] == "ready"


def test_predict_requires_api_key_when_enabled() -> None:
    settings = Settings(require_api_key=True, model_service_api_key="secret-key")
    app = create_app(settings, predictor=DummyPredictor(is_ready=True))
    client = app.test_client()

    response = client.post("/predict")

    assert response.status_code == 401


def test_predict_returns_error_when_file_is_missing() -> None:
    app = create_app(Settings(), predictor=DummyPredictor(is_ready=True))
    client = app.test_client()

    response = client.post("/predict", data={}, content_type="multipart/form-data")

    assert response.status_code == 400


def test_predict_returns_mask_payload() -> None:
    app = create_app(Settings(), predictor=DummyPredictor(is_ready=True))
    client = app.test_client()

    response = client.post(
        "/predict",
        data={"file": (make_test_image(), "sample.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["message"] == "Segmentation generated successfully."
    assert "mask_png_base64" in payload
