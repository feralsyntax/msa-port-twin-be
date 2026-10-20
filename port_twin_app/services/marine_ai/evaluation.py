"""Evaluation utilities for marine FNO forecasting."""

from dataclasses import dataclass
from pathlib import Path

import torch

from .contracts import MarinePredictionRequest, MarinePredictionResult
from .dataset import MarineDatasetSample
from .model import (
    MODEL_VERSION,
    MarineFNO,
    load_checkpoint,
    predict,
)

OUTPUT_FIELDS = (
    "water_level_m",
    "current_u_ms",
    "current_v_ms",
)


@dataclass(frozen=True)
class RegressionMetrics:
    """Regression metrics aggregated over forecast cells and time steps."""

    mae: float
    rmse: float


@dataclass(frozen=True)
class ModelEvaluationResult:
    """Summary of model and baseline performance on evaluation samples."""

    model_version: str
    samples_evaluated: int
    fno_metrics: dict[str, RegressionMetrics]
    baseline_metrics: dict[str, RegressionMetrics]
    data_sources: tuple[str, ...]
    evaluation_note: str = (
        "Results apply only to the supplied evaluation samples. "
        "Synthetic fixtures do not establish real-world forecasting accuracy."
    )


def _result_tensor(result: MarinePredictionResult) -> torch.Tensor:
    """Convert prediction fields to [steps, 3, ny, nx]."""

    return torch.tensor(
        [
            result.water_level_m,
            result.current_u_ms,
            result.current_v_ms,
        ],
        dtype=torch.float32,
    ).permute(1, 0, 2, 3)


def _baseline_prediction(
    request: MarinePredictionRequest,
) -> MarinePredictionResult:
    """Create a simple persistence/zero-current baseline.

    Water level remains equal to the initial water-level grid at every
    forecast step. Both current components are assumed to be zero because
    the request contract does not provide initial current grids.
    """

    steps = request.forecast.steps
    nx = request.grid.nx
    ny = request.grid.ny

    initial_level = torch.as_tensor(
        request.initial_water_level_m,
        dtype=torch.float32,
    )

    if tuple(initial_level.shape) != (ny, nx):
        raise ValueError("initial_water_level_m does not match the request grid.")

    if not torch.isfinite(initial_level).all():
        raise ValueError("initial_water_level_m contains non-finite values.")

    zero_current = torch.zeros(
        steps,
        ny,
        nx,
        dtype=torch.float32,
    )

    return MarinePredictionResult(
        water_level_m=[initial_level.tolist() for _ in range(steps)],
        current_u_ms=zero_current.tolist(),
        current_v_ms=zero_current.tolist(),
        model_version="persistence-zero-current-baseline",
    )


def _calculate_metrics(
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
) -> dict[str, RegressionMetrics]:
    """Calculate MAE and RMSE for each output variable."""

    if not predictions or len(predictions) != len(targets):
        raise ValueError(
            "Predictions and targets must be non-empty and have equal lengths."
        )

    errors = torch.cat(
        [prediction - target for prediction, target in zip(predictions, targets)],
        dim=0,
    )

    if not torch.isfinite(errors).all():
        raise ValueError("Evaluation errors contain non-finite values.")

    metrics = {}

    for channel, field_name in enumerate(OUTPUT_FIELDS):
        field_errors = errors[:, :, channel, :, :]

        mae = field_errors.abs().mean().item()
        rmse = field_errors.square().mean().sqrt().item()

        metrics[field_name] = RegressionMetrics(
            mae=mae,
            rmse=rmse,
        )

    return metrics


def evaluate_model(
    model: MarineFNO,
    samples: tuple[MarineDatasetSample, ...],
    device: str = "cpu",
    target_mean: torch.Tensor | None = None,
    target_std: torch.Tensor | None = None,
) -> ModelEvaluationResult:
    """Evaluate an FNO against a baseline on supplied samples.

    This function does not train the model, fit normalization statistics,
    or modify model parameters. Supply target_mean and target_std when the
    model was trained using normalized targets.
    """

    if not isinstance(model, MarineFNO):
        raise TypeError("model must be a MarineFNO instance.")

    if not samples:
        raise ValueError("Evaluation samples cannot be empty.")

    if (target_mean is None) != (target_std is None):
        raise ValueError(
            "target_mean and target_std must either both be supplied "
            "or both be omitted."
        )

    fno_predictions = []
    baseline_predictions = []
    targets = []
    sources = set()

    for sample in samples:
        result = predict(
            sample.request,
            model,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
        )
        result.validate_against(sample.request)

        baseline = _baseline_prediction(sample.request)
        baseline.validate_against(sample.request)
        sample.target.validate_against(sample.request)

        fno_predictions.append(_result_tensor(result))
        baseline_predictions.append(_result_tensor(baseline))
        targets.append(_result_tensor(sample.target))

        sources.add(sample.source)

    return ModelEvaluationResult(
        model_version=MODEL_VERSION,
        samples_evaluated=len(samples),
        fno_metrics=_calculate_metrics(fno_predictions, targets),
        baseline_metrics=_calculate_metrics(baseline_predictions, targets),
        data_sources=tuple(sorted(sources)),
    )


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    samples: tuple[MarineDatasetSample, ...],
    device: str = "cpu",
) -> ModelEvaluationResult:
    """Load a trained checkpoint and evaluate it on supplied samples."""

    model, checkpoint = load_checkpoint(
        checkpoint_path,
        device=device,
    )

    target_mean = checkpoint.get("target_mean")
    target_std = checkpoint.get("target_std")

    if (target_mean is None) != (target_std is None):
        raise ValueError("Checkpoint must contain both target_mean and target_std.")

    result = evaluate_model(
        model,
        samples,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
    )

    return ModelEvaluationResult(
        model_version=checkpoint.get("model_version", result.model_version),
        samples_evaluated=result.samples_evaluated,
        fno_metrics=result.fno_metrics,
        baseline_metrics=result.baseline_metrics,
        data_sources=result.data_sources,
        evaluation_note=result.evaluation_note,
    )
