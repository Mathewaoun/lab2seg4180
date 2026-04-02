from __future__ import annotations

from flask import Flask, jsonify, request

from model_service.config import Settings
from model_service.inference import SegmentationPredictor


def create_app(settings: Settings | None = None, predictor: SegmentationPredictor | None = None) -> Flask:
    settings = settings or Settings.from_env()
    predictor = predictor or SegmentationPredictor(settings)

    app = Flask(__name__)
    app.config["SETTINGS"] = settings
    app.config["PREDICTOR"] = predictor

    @app.get("/")
    def index():
        return jsonify(
            {
                "service": "house-segmentation-api",
                "endpoint": "/predict",
                "auth": "X-API-Key header required when REQUIRE_API_KEY=true",
            }
        )

    @app.get("/health")
    def health():
        current_predictor = app.config["PREDICTOR"]
        return jsonify(
            {
                "status": "ready" if current_predictor.is_ready else "degraded",
                "model_loaded": current_predictor.is_ready,
                "weights_path": str(settings.model_weights_path),
                "api_key_protected": settings.require_api_key,
                "message": current_predictor.status_message,
            }
        )

    @app.post("/predict")
    def predict():
        if settings.require_api_key and request.headers.get("X-API-Key") != settings.model_service_api_key:
            return jsonify({"error": "Unauthorized. Provide the correct X-API-Key header."}), 401

        current_predictor = app.config["PREDICTOR"]
        if not current_predictor.is_ready:
            return (
                jsonify(
                    {
                        "error": "Model weights are not available.",
                        "weights_path": str(settings.model_weights_path),
                        "message": current_predictor.status_message,
                    }
                ),
                503,
            )

        image_file = request.files.get("file")
        if image_file is None or not image_file.filename:
            return jsonify({"error": "Upload an image file using the 'file' form field."}), 400

        ground_truth = request.files.get("ground_truth")
        result = current_predictor.predict(
            image_bytes=image_file.read(),
            ground_truth_bytes=ground_truth.read() if ground_truth else None,
        )
        return jsonify(result)

    return app


app = create_app()


if __name__ == "__main__":
    runtime_settings = app.config["SETTINGS"]
    app.run(host=runtime_settings.app_host, port=runtime_settings.app_port)
