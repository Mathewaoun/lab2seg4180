from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_host: str = "0.0.0.0"
    app_port: int = 5000
    model_weights_path: str = "artifacts/checkpoints/best_model.pt"
    prediction_size: int = 256
    prediction_threshold: float = 0.5
    require_api_key: bool = False
    model_service_api_key: str = ""
    model_device: str = "cpu"
    data_dir: str = "data/processed"
    artifacts_dir: str = "artifacts"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            app_host=os.getenv("APP_HOST", "0.0.0.0"),
            app_port=int(os.getenv("APP_PORT", "5000")),
            model_weights_path=os.getenv(
                "MODEL_WEIGHTS_PATH",
                "artifacts/checkpoints/best_model.pt",
            ),
            prediction_size=int(os.getenv("PREDICTION_SIZE", "256")),
            prediction_threshold=float(os.getenv("PREDICTION_THRESHOLD", "0.5")),
            require_api_key=_env_bool("REQUIRE_API_KEY", False),
            model_service_api_key=os.getenv("MODEL_SERVICE_API_KEY", ""),
            model_device=os.getenv("MODEL_DEVICE", "cpu"),
            data_dir=os.getenv("DATA_DIR", "data/processed"),
            artifacts_dir=os.getenv("ARTIFACTS_DIR", "artifacts"),
        )
