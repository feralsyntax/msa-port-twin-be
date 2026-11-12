"""
Reusable inference service for marine FNO predictions.

What it does:
- Loads a trained checkpoint once when the service is initialized.
- Accepts a MarinePredictionRequest and returns a MarinePredictionResult.
- Uses the checkpoint's normalization statistics when available.
- Supports absolute and residual prediction checkpoints.
- Reports checkpoint-loading and inference errors clearly.
"""

from pathlib import Path

from .contracts import MarinePredictionRequest, MarinePredictionResult
from .model import (
    PREDICTION_MODE_ABSOLUTE,
    load_checkpoint,
    predict,
)


class MarineInferenceService:
    """Load a trained model once and serve marine prediction requests."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
    ):
        self.device = device

        self.model, checkpoint = load_checkpoint(
            checkpoint_path,
            device=device,
        )

        self.model_version = checkpoint["model_version"]
        self.prediction_mode = checkpoint.get(
            "prediction_mode",
            PREDICTION_MODE_ABSOLUTE,
        )

        self.target_mean = checkpoint.get("target_mean")
        self.target_std = checkpoint.get("target_std")
        self.input_mean = checkpoint.get("input_mean")
        self.input_std = checkpoint.get("input_std")

        if (self.input_mean is None) != (self.input_std is None):
            raise ValueError("Checkpoint must contain both input_mean and input_std.")

        if (self.target_mean is None) != (self.target_std is None):
            raise ValueError("Checkpoint must contain both target_mean and target_std.")

    def predict(
        self,
        request: MarinePredictionRequest,
    ) -> MarinePredictionResult:
        """Generate and validate predictions for a marine request."""

        if not isinstance(request, MarinePredictionRequest):
            raise TypeError("request must be a MarinePredictionRequest instance.")

        result = predict(
            request,
            self.model,
            device=self.device,
            target_mean=self.target_mean,
            target_std=self.target_std,
            input_mean=self.input_mean,
            input_std=self.input_std,
            prediction_mode=self.prediction_mode,
        )

        result.validate_against(request)

        return MarinePredictionResult(
            water_level_m=result.water_level_m,
            current_u_ms=result.current_u_ms,
            current_v_ms=result.current_v_ms,
            model_version=self.model_version,
        )
