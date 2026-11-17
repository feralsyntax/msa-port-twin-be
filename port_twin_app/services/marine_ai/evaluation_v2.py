"""Evaluation utilities for forcing-conditioned marine FNO v2."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .model_v2 import MODEL_VERSION, MarineFNOV2, load_checkpoint
from .preparation_v2 import MarineFNOV2Sample
from .training_v2 import build_input_tensor

OUTPUT_FIELDS = (
    "water_level_m",
    "current_u_ms",
    "current_v_ms",
)


@dataclass(frozen=True)
class RegressionMetrics:
    """MAE and RMSE aggregated over valid water cells and forecast steps."""

    mae: float
    rmse: float


@dataclass(frozen=True)
class EvaluationV2Result:
    """Held-out comparison of FNO v2 and simple forecasting baselines."""

    model_version: str
    samples_evaluated: int
    fno_metrics: dict[str, RegressionMetrics]
    persistence_metrics: dict[str, RegressionMetrics]
    harmonic_tide_metrics: dict[str, RegressionMetrics]
    fno_beats_persistence: bool
    promotion_eligible: bool
    out_of_domain_models: tuple[str, ...] = ("fno-0.4.0",)
    evaluation_note: str = (
        "Evaluation uses simulated v2 held-out scenarios only. "
        "The harmonic-tide baseline is spatially uniform and keeps "
        "currents persistent. Results do not establish real-world accuracy."
    )


def _validate_sample(sample: MarineFNOV2Sample) -> tuple[np.ndarray, ...]:
    """Validate the arrays required for evaluation."""

    mask = np.asarray(sample.water_mask, dtype=bool)
    initial = np.asarray(sample.initial_state, dtype=np.float32)
    target = np.asarray(sample.target, dtype=np.float32)

    if mask.ndim != 2 or not mask.any():
        raise ValueError(f"Sample {sample.sample_id!r} has an invalid water mask.")

    ny, nx = mask.shape

    if initial.shape != (3, ny, nx):
        raise ValueError("Initial state does not match the water mask.")

    if target.shape != (6, 3, ny, nx):
        raise ValueError("Target does not have shape [6, 3, ny, nx].")

    if not np.isfinite(initial[:, mask]).all():
        raise ValueError("Initial water state contains non-finite values.")

    if not np.isfinite(target[:, :, mask]).all():
        raise ValueError("Water-cell targets contain non-finite values.")

    if not isinstance(sample.forcing, dict):
        raise ValueError("Sample forcing must be a dictionary.")  # noqa: TRY004

    for key in (
        "tide_amplitude_m",
        "tide_phase_radians",
        "tide_period_seconds",
    ):
        value = sample.forcing.get(key)

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Invalid forcing value: {key}.")  # noqa: TRY004

        if not np.isfinite(value):
            raise ValueError(f"Non-finite forcing value: {key}.")

    if sample.forcing["tide_period_seconds"] <= 0:
        raise ValueError("Tide period must be positive.")

    return mask, initial, target


def _metrics(
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
    masks: list[np.ndarray],
) -> dict[str, RegressionMetrics]:
    """Calculate per-variable metrics using water cells only."""

    if not predictions or len(predictions) != len(targets):
        raise ValueError("Predictions and targets must be non-empty and aligned.")

    if len(predictions) != len(masks):
        raise ValueError("Predictions and masks must have equal lengths.")

    results = {}

    for channel, field_name in enumerate(OUTPUT_FIELDS):
        errors = []

        for prediction, target, mask in zip(predictions, targets, masks):
            if prediction.shape != target.shape:
                raise ValueError("Prediction and target shapes do not match.")

            if prediction.shape != (6, 3, *mask.shape):
                raise ValueError("Prediction does not match the sample grid.")

            field_errors = (
                prediction[:, channel, :, :][:, mask]
                - (target[:, channel, :, :][:, mask])
            )

            if not np.isfinite(field_errors).all():
                raise ValueError("Evaluation errors contain non-finite values.")

            errors.append(field_errors.reshape(-1))

        values = np.concatenate(errors)

        results[field_name] = RegressionMetrics(
            mae=float(np.mean(np.abs(values))),
            rmse=float(np.sqrt(np.mean(np.square(values)))),
        )

    return results


def _persistence_prediction(initial: np.ndarray) -> np.ndarray:
    """Repeat the initial physical state for six forecast hours."""

    return np.broadcast_to(initial[None, ...], (6, 3, *initial.shape[1:])).copy()


def _harmonic_tide_prediction(sample: MarineFNOV2Sample) -> np.ndarray:
    """Predict spatially uniform harmonic water level and persistent currents."""

    _, initial, _ = _validate_sample(sample)

    forcing = sample.forcing
    amplitude = float(forcing["tide_amplitude_m"])
    phase = float(forcing["tide_phase_radians"])
    period = float(forcing["tide_period_seconds"])

    result = _persistence_prediction(initial)

    for step in range(6):
        forecast_time = sample.start_time_seconds + (step + 1) * 3600.0
        water_level = amplitude * np.sin((2.0 * np.pi * forecast_time / period) + phase)
        result[step, 0, :, :] = water_level

    return result


def _predict_fno(
    model: MarineFNOV2,
    sample: MarineFNOV2Sample,
    device: str,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> np.ndarray:
    """Run v2 inference and return predictions in physical units."""

    inputs = build_input_tensor(sample, device=device)

    inputs = (inputs - input_mean) / input_std

    with torch.no_grad():
        outputs = model(inputs)

    outputs = outputs * target_std + target_mean

    if not torch.isfinite(outputs).all():
        raise ValueError("FNO v2 produced non-finite predictions.")

    prediction = outputs[0].detach().cpu().numpy()

    if prediction.shape != (6, 3, *sample.water_mask.shape):
        raise ValueError("FNO v2 output shape does not match the sample.")

    return prediction


def evaluate_model_v2(
    model: MarineFNOV2,
    samples: tuple[MarineFNOV2Sample, ...],
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: str = "cpu",
) -> EvaluationV2Result:
    """Evaluate FNO v2 and both baselines on the same held-out samples."""

    if not isinstance(model, MarineFNOV2):
        raise TypeError("model must be a MarineFNOV2 instance.")

    if not samples:
        raise ValueError("Evaluation samples cannot be empty.")

    model = model.to(device)
    model.eval()

    input_mean = torch.as_tensor(input_mean, dtype=torch.float32, device=device)
    input_std = torch.as_tensor(input_std, dtype=torch.float32, device=device)
    target_mean = torch.as_tensor(target_mean, dtype=torch.float32, device=device)
    target_std = torch.as_tensor(target_std, dtype=torch.float32, device=device)

    expected_shapes = (
        (input_mean, (1, 13, 1, 1), "input_mean"),
        (input_std, (1, 13, 1, 1), "input_std"),
        (target_mean, (1, 1, 3, 1, 1), "target_mean"),
        (target_std, (1, 1, 3, 1, 1), "target_std"),
    )

    for tensor, expected_shape, name in expected_shapes:
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}.")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains non-finite values.")

    if torch.any(input_std <= 0) or torch.any(target_std <= 0):
        raise ValueError("Normalization standard deviations must be positive.")

    fno_predictions = []
    persistence_predictions = []
    harmonic_predictions = []
    targets = []
    masks = []

    for sample in samples:
        mask, initial, target = _validate_sample(sample)

        fno_predictions.append(
            _predict_fno(
                model,
                sample,
                device,
                input_mean,
                input_std,
                target_mean,
                target_std,
            )
        )
        persistence_predictions.append(_persistence_prediction(initial))
        harmonic_predictions.append(_harmonic_tide_prediction(sample))
        targets.append(target)
        masks.append(mask)

    fno_metrics = _metrics(fno_predictions, targets, masks)
    persistence_metrics = _metrics(persistence_predictions, targets, masks)
    harmonic_metrics = _metrics(harmonic_predictions, targets, masks)

    fno_beats_persistence = all(
        fno_metrics[field].mae < persistence_metrics[field].mae
        for field in OUTPUT_FIELDS
    )

    return EvaluationV2Result(
        model_version=MODEL_VERSION,
        samples_evaluated=len(samples),
        fno_metrics=fno_metrics,
        persistence_metrics=persistence_metrics,
        harmonic_tide_metrics=harmonic_metrics,
        fno_beats_persistence=fno_beats_persistence,
        promotion_eligible=fno_beats_persistence,
    )


def evaluate_checkpoint_v2(
    checkpoint_path: str | Path,
    samples: tuple[MarineFNOV2Sample, ...],
    device: str = "cpu",
) -> EvaluationV2Result:
    """Load a trained v2 checkpoint and evaluate its held-out performance."""

    model, checkpoint = load_checkpoint(checkpoint_path, device=device)

    required_statistics = (
        "input_mean",
        "input_std",
        "target_mean",
        "target_std",
    )

    missing = [key for key in required_statistics if key not in checkpoint]
    if missing:
        raise ValueError(f"Checkpoint is missing normalization statistics: {missing}.")

    return evaluate_model_v2(
        model,
        samples,
        input_mean=checkpoint["input_mean"],
        input_std=checkpoint["input_std"],
        target_mean=checkpoint["target_mean"],
        target_std=checkpoint["target_std"],
        device=device,
    )
