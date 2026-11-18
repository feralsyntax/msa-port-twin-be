"""Training utilities for the forcing-conditioned marine FNO v2."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .model_v2 import (
    FORECAST_STEPS,
    INPUT_CHANNELS,
    INPUT_STATISTICS_SHAPE,
    TARGET_STATISTICS_SHAPE,
    MarineFNOV2,
    load_checkpoint,
    save_checkpoint,
)
from .preparation_v2 import MarineFNOV2Sample, MarineFNOV2Split


@dataclass(frozen=True)
class TrainingV2Config:
    """Configuration for v2 supervised and physics-informed training."""

    epochs: int = 20
    learning_rate: float = 1e-3
    device: str = "cpu"
    checkpoint_path: str = "artifacts/marine_fno_v2.pt"
    seed: int = 42
    physics_loss_weight: float = 0.1
    continuity_residual_scale_ms: float = 1e-4

    def __post_init__(self):
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int):
            raise ValueError("epochs must be a positive integer.")  # noqa: TRY004
        if self.epochs < 1:
            raise ValueError("epochs must be a positive integer.")

        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate must be positive.")

        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string.")

        if (
            not isinstance(self.checkpoint_path, str)
            or not self.checkpoint_path.strip()
        ):
            raise ValueError("checkpoint_path must be a non-empty string.")

        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be a non-negative integer.")  # noqa: TRY004
        if self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")

        if (
            isinstance(self.physics_loss_weight, bool)
            or not isinstance(self.physics_loss_weight, (int, float))
            or self.physics_loss_weight < 0
        ):
            raise ValueError("physics_loss_weight must be non-negative.")

        if (
            isinstance(self.continuity_residual_scale_ms, bool)
            or not isinstance(
                self.continuity_residual_scale_ms,
                (int, float),
            )
            or self.continuity_residual_scale_ms <= 0
        ):
            raise ValueError("continuity_residual_scale_ms must be positive.")


@dataclass(frozen=True)
class TrainingV2Result:
    """Summary of a completed v2 training run."""

    model_version: str
    epochs_completed: int
    training_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]
    training_supervised_losses: tuple[float, ...]
    training_continuity_losses: tuple[float, ...]
    validation_supervised_losses: tuple[float, ...]
    validation_continuity_losses: tuple[float, ...]
    best_validation_loss: float
    best_epoch: int
    checkpoint_path: str


def _sample_arrays(sample: MarineFNOV2Sample):
    """Validate sample arrays and return float32 arrays."""

    mask = np.asarray(sample.water_mask, dtype=bool)
    initial = np.asarray(sample.initial_state, dtype=np.float32)
    bathymetry = np.asarray(sample.bathymetry_m, dtype=np.float32)
    target = np.asarray(sample.target, dtype=np.float32)
    easting = np.asarray(sample.easting_m, dtype=np.float64)
    northing = np.asarray(sample.northing_m, dtype=np.float64)

    if mask.ndim != 2 or not mask.any():
        raise ValueError(f"Sample {sample.sample_id!r} has an invalid water mask.")

    ny, nx = mask.shape

    if initial.shape != (3, ny, nx):
        raise ValueError(f"Sample {sample.sample_id!r} has an invalid initial state.")

    if bathymetry.shape != (ny, nx):
        raise ValueError(f"Sample {sample.sample_id!r} has invalid bathymetry.")

    if target.shape != (FORECAST_STEPS, 3, ny, nx):
        raise ValueError(f"Sample {sample.sample_id!r} has invalid target shape.")

    if easting.shape != (nx,) or northing.shape != (ny,):
        raise ValueError(f"Sample {sample.sample_id!r} has invalid grid coordinates.")

    if (
        not np.isfinite(initial[:, mask]).all()
        or not np.isfinite(bathymetry[mask]).all()
        or not np.isfinite(target[:, :, mask]).all()
    ):
        raise ValueError(f"Sample {sample.sample_id!r} has non-finite water values.")

    if np.any(bathymetry[mask] <= 0):
        raise ValueError("Water-cell bathymetry must be positive downward.")

    if (
        not np.isfinite(easting).all()
        or not np.isfinite(northing).all()
        or np.any(np.diff(easting) <= 0)
        or np.any(np.diff(northing) <= 0)
    ):
        raise ValueError("Grid coordinates must be finite and increasing.")

    if nx < 3 or ny < 3:
        raise ValueError("Continuity loss requires at least a 3x3 grid.")

    if not isinstance(sample.forcing, dict):
        raise ValueError("Sample forcing must be a dictionary.")  # noqa: TRY004

    required_forcing = (
        "wind_u_ms",
        "wind_v_ms",
        "tide_amplitude_m",
        "tide_phase_radians",
        "tide_period_seconds",
        "tidal_boundary",
    )

    for key in required_forcing:
        if key not in sample.forcing:
            raise ValueError(f"Sample forcing is missing {key}.")

    for key in required_forcing[:-1]:
        value = sample.forcing[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Invalid numeric forcing value: {key}.")  # noqa: TRY004
        if not np.isfinite(value):
            raise ValueError(f"Non-finite forcing value: {key}.")

    if sample.forcing["tide_period_seconds"] <= 0:
        raise ValueError("Tide period must be positive.")

    if sample.forcing["tidal_boundary"] != "east":
        raise ValueError("FNO v2 requires an eastern tidal boundary.")

    if (
        not np.isclose(
            np.diff(easting),
            np.diff(easting)[0],
        ).all()
        or not np.isclose(
            np.diff(northing),
            np.diff(northing)[0],
        ).all()
    ):
        raise ValueError("Continuity loss requires regularly spaced coordinates.")

    return mask, initial, bathymetry, target, easting, northing


def build_input_tensor(
    sample: MarineFNOV2Sample,
    device: str = "cpu",
) -> torch.Tensor:
    """Assemble the documented 13 input channels as [1, 13, ny, nx]."""

    mask, initial, bathymetry, _, easting, northing = _sample_arrays(sample)

    ny, nx = mask.shape

    # Invalid land values are replaced with zero; the mask remains explicit.
    bathymetry = np.where(mask, bathymetry, 0.0)
    initial = np.where(mask[None, :, :], initial, 0.0)

    easting_normalized = (
        2.0 * (easting - easting.min()) / (easting.max() - easting.min()) - 1.0
    )
    northing_normalized = (
        2.0 * (northing - northing.min()) / (northing.max() - northing.min()) - 1.0
    )

    east_grid, north_grid = np.meshgrid(
        easting_normalized,
        northing_normalized,
    )

    forcing = sample.forcing
    phase = float(forcing["tide_phase_radians"])

    channels = np.stack(
        (
            bathymetry,
            initial[0],
            initial[1],
            initial[2],
            mask.astype(np.float32),
            east_grid.astype(np.float32),
            north_grid.astype(np.float32),
            np.full((ny, nx), forcing["wind_u_ms"], dtype=np.float32),
            np.full((ny, nx), forcing["wind_v_ms"], dtype=np.float32),
            np.full(
                (ny, nx),
                forcing["tide_amplitude_m"],
                dtype=np.float32,
            ),
            np.full((ny, nx), np.sin(phase), dtype=np.float32),
            np.full((ny, nx), np.cos(phase), dtype=np.float32),
            np.broadcast_to(
                np.arange(nx)[None, :] == nx - 1,
                (ny, nx),
            ).astype(np.float32),
        ),
        axis=0,
    )

    if channels.shape != (INPUT_CHANNELS, ny, nx):
        raise ValueError("Assembled v2 inputs do not match the 13-channel contract.")

    if not np.isfinite(channels).all():
        raise ValueError("Assembled v2 inputs contain non-finite values.")

    return torch.as_tensor(
        channels,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)


def _target_tensor(
    sample: MarineFNOV2Sample,
    device: str,
) -> torch.Tensor:
    """Return targets as [1, 6, 3, ny, nx], replacing land NaNs with zero."""

    mask, _, _, target, _, _ = _sample_arrays(sample)
    target = np.where(mask[None, None, :, :], target, 0.0)

    return torch.as_tensor(
        target,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)


def _fit_input_normalization(
    samples: tuple[MarineFNOV2Sample, ...],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit channel statistics using training samples and water cells only."""

    if not samples:
        raise ValueError("Training samples cannot be empty.")

    values = []

    for sample in samples:
        inputs = build_input_tensor(sample, device=device)[0]
        mask = torch.as_tensor(
            sample.water_mask,
            dtype=torch.bool,
            device=device,
        )
        values.append(inputs[:, mask])

    combined = torch.cat(values, dim=1)

    mean = combined.mean(dim=1).reshape(INPUT_STATISTICS_SHAPE)
    std = combined.std(dim=1, unbiased=False).reshape(INPUT_STATISTICS_SHAPE)

    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("Training inputs contain non-finite values.")

    std = torch.where(std < 1e-6, torch.ones_like(std), std)

    return mean, std


def _fit_target_normalization(
    samples: tuple[MarineFNOV2Sample, ...],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit output statistics using training targets over water cells only."""

    if not samples:
        raise ValueError("Training samples cannot be empty.")

    values = []

    for sample in samples:
        targets = _target_tensor(sample, device=device)[0]
        mask = torch.as_tensor(
            sample.water_mask,
            dtype=torch.bool,
            device=device,
        )
        values.append(targets[:, :, mask])

    # [all sample-steps, 3, all water cells]
    combined = torch.cat(values, dim=0)

    mean = combined.mean(dim=(0, 2)).reshape(TARGET_STATISTICS_SHAPE)
    std = combined.std(
        dim=(0, 2),
        unbiased=False,
    ).reshape(TARGET_STATISTICS_SHAPE)

    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("Training targets contain non-finite values.")

    std = torch.where(std < 1e-6, torch.ones_like(std), std)

    return mean, std


def _masked_supervised_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Calculate normalized MSE over valid water cells only."""

    if predictions.shape != targets.shape:
        raise ValueError("Prediction and target shapes do not match.")

    expanded_mask = mask[:, None, None, :, :].expand_as(predictions)

    return F.mse_loss(
        predictions[expanded_mask],
        targets[expanded_mask],
    )


def _continuity_loss(
    physical_predictions: torch.Tensor,
    initial_state: torch.Tensor,
    bathymetry: torch.Tensor,
    mask: torch.Tensor,
    easting: torch.Tensor,
    northing: torch.Tensor,
    residual_scale_ms: float = 1e-4,
) -> torch.Tensor:
    """Penalize the shallow-water continuity residual in interior wet cells.

    Land-cell values are sanitized before arithmetic because bathymetry and
    initial-state arrays may contain NaNs outside the water mask.
    """

    if physical_predictions.ndim != 5:
        raise ValueError("Physical predictions must have five dimensions.")

    if physical_predictions.shape[1:3] != (FORECAST_STEPS, 3):
        raise ValueError("Physical predictions have an invalid output shape.")

    dx = easting[1] - easting[0]
    dy = northing[1] - northing[0]

    if dx <= 0 or dy <= 0:
        raise ValueError("Grid spacing must be positive.")

    valid = (
        mask[1:-1, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
    )

    if not valid.any():
        return physical_predictions.sum() * 0.0

    # Sanitize land cells BEFORE arithmetic to prevent NaN propagation
    # through multiplication and its backward gradients.
    safe_bathymetry = torch.where(
        mask,
        bathymetry,
        torch.zeros_like(bathymetry),
    )

    previous_eta = torch.where(
        mask,
        initial_state[0],
        torch.zeros_like(initial_state[0]),
    )

    losses = []

    for step in range(FORECAST_STEPS):
        eta = physical_predictions[0, step, 0]
        current_u = physical_predictions[0, step, 1]
        current_v = physical_predictions[0, step, 2]

        # Predictions on land are irrelevant to the continuity equation.
        safe_eta = torch.where(mask, eta, torch.zeros_like(eta))
        safe_current_u = torch.where(mask, current_u, torch.zeros_like(current_u))
        safe_current_v = torch.where(mask, current_v, torch.zeros_like(current_v))

        depth = torch.clamp_min(safe_bathymetry + safe_eta, 0.0)

        flux_x = depth * safe_current_u
        flux_y = depth * safe_current_v

        divergence_x = (flux_x[1:-1, 2:] - flux_x[1:-1, :-2]) / (2.0 * dx)

        divergence_y = (flux_y[2:, 1:-1] - flux_y[:-2, 1:-1]) / (2.0 * dy)

        time_derivative = (safe_eta[1:-1, 1:-1] - previous_eta[1:-1, 1:-1]) / 3600.0

        residual = time_derivative + divergence_x + divergence_y

        normalized_residual = residual[valid] / residual_scale_ms
        losses.append(normalized_residual.square().mean())

        previous_eta = safe_eta

    return torch.stack(losses).mean()


def _sample_loss(
    model: MarineFNOV2,
    sample: MarineFNOV2Sample,
    device: str,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    physics_loss_weight: float,
    continuity_residual_scale_ms: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, supervised and continuity losses for one sample."""

    inputs = build_input_tensor(sample, device=device)
    targets = _target_tensor(sample, device=device)

    mask = torch.as_tensor(
        sample.water_mask,
        dtype=torch.bool,
        device=device,
    ).unsqueeze(0)

    normalized_inputs = (inputs - input_mean) / input_std
    normalized_targets = (targets - target_mean) / target_std

    normalized_predictions = model(normalized_inputs)

    if not torch.isfinite(normalized_predictions).all():
        raise ValueError(
            f"Model produced non-finite predictions for {sample.sample_id!r}."
        )

    supervised_loss = _masked_supervised_loss(
        normalized_predictions,
        normalized_targets,
        mask,
    )

    # Inverse normalization before evaluating the physical equation.
    physical_predictions = normalized_predictions * target_std + target_mean

    _, initial, bathymetry, _, easting, northing = _sample_arrays(sample)

    initial_tensor = torch.as_tensor(
        initial,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    bathymetry_tensor = torch.as_tensor(
        bathymetry,
        dtype=torch.float32,
        device=device,
    )

    easting_tensor = torch.as_tensor(
        easting,
        dtype=torch.float32,
        device=device,
    )
    northing_tensor = torch.as_tensor(
        northing,
        dtype=torch.float32,
        device=device,
    )

    continuity_loss = _continuity_loss(
        physical_predictions,
        initial_tensor[0],
        bathymetry_tensor,
        mask[0],
        easting_tensor,
        northing_tensor,
        residual_scale_ms=continuity_residual_scale_ms,
    )

    total_loss = supervised_loss + physics_loss_weight * continuity_loss

    return total_loss, supervised_loss, continuity_loss


def _evaluate(
    model: MarineFNOV2,
    samples: tuple[MarineFNOV2Sample, ...],
    device: str,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    config: TrainingV2Config,
) -> tuple[float, float, float]:
    """Calculate mean combined validation loss without updating weights."""

    if not samples:
        raise ValueError("Validation samples cannot be empty.")

    model.eval()
    total_losses = []
    supervised_losses = []
    continuity_losses = []

    with torch.no_grad():
        for sample in samples:
            total, supervised, continuity = _sample_loss(
                model,
                sample,
                device,
                input_mean,
                input_std,
                target_mean,
                target_std,
                config.physics_loss_weight,
                config.continuity_residual_scale_ms,
            )

            if not all(
                torch.isfinite(loss) for loss in (total, supervised, continuity)
            ):
                raise ValueError(
                    f"Non-finite validation loss for {sample.sample_id!r}."
                )

            total_losses.append(total.item())
            supervised_losses.append(supervised.item())
            continuity_losses.append(continuity.item())

    return (
        sum(total_losses) / len(total_losses),
        sum(supervised_losses) / len(supervised_losses),
        sum(continuity_losses) / len(continuity_losses),
    )


def train_model_v2(
    model: MarineFNOV2,
    dataset: MarineFNOV2Split,
    config: TrainingV2Config | None = None,
) -> TrainingV2Result:
    """Train v2 using training-only normalization and continuity regularization."""

    config = config or TrainingV2Config()

    if not isinstance(model, MarineFNOV2):
        raise TypeError("model must be a MarineFNOV2 instance.")

    if not isinstance(dataset, MarineFNOV2Split):
        raise TypeError("dataset must be a MarineFNOV2Split instance.")

    for partition in ("train", "validation", "test"):
        if not getattr(dataset, partition):
            raise ValueError(f"The {partition} partition cannot be empty.")

    torch.manual_seed(config.seed)

    device = torch.device(config.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")

    model = model.to(device)

    input_mean, input_std = _fit_input_normalization(
        dataset.train,
        str(device),
    )
    target_mean, target_std = _fit_target_normalization(
        dataset.train,
        str(device),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    training_losses = []
    validation_losses = []
    training_supervised_losses = []
    training_continuity_losses = []
    validation_supervised_losses = []
    validation_continuity_losses = []
    best_validation_loss = float("inf")
    best_epoch = 0
    checkpoint_path = Path(config.checkpoint_path)

    # Temporary: capture forward operations responsible for invalid gradients.
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses = []
        epoch_supervised_losses = []
        epoch_continuity_losses = []

        for sample in dataset.train:
            optimizer.zero_grad(set_to_none=True)

            total_loss, supervised_loss, continuity_loss = _sample_loss(
                model,
                sample,
                str(device),
                input_mean,
                input_std,
                target_mean,
                target_std,
                config.physics_loss_weight,
                config.continuity_residual_scale_ms,
            )

            if not torch.isfinite(total_loss):
                raise ValueError(f"Non-finite training loss for {sample.sample_id!r}.")

            total_loss.backward()

            if not all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise ValueError(f"Non-finite gradients for {sample.sample_id!r}.")

            optimizer.step()

            epoch_losses.append(total_loss.item())
            epoch_supervised_losses.append(supervised_loss.item())
            epoch_continuity_losses.append(continuity_loss.item())

        training_loss = sum(epoch_losses) / len(epoch_losses)
        training_supervised_loss = sum(epoch_supervised_losses) / len(
            epoch_supervised_losses
        )
        training_continuity_loss = sum(epoch_continuity_losses) / len(
            epoch_continuity_losses
        )

        (
            validation_loss,
            validation_supervised_loss,
            validation_continuity_loss,
        ) = _evaluate(
            model,
            dataset.validation,
            str(device),
            input_mean,
            input_std,
            target_mean,
            target_std,
            config,
        )

        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
        training_supervised_losses.append(training_supervised_loss)
        training_continuity_losses.append(training_continuity_loss)
        validation_supervised_losses.append(validation_supervised_loss)
        validation_continuity_losses.append(validation_continuity_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch

            save_checkpoint(
                model,
                checkpoint_path,
                optimizer=optimizer,
                epoch=epoch,
                input_mean=input_mean,
                input_std=input_std,
                target_mean=target_mean,
                target_std=target_std,
            )

            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            checkpoint["training"] = {
                "seed": config.seed,
                "epochs_completed": epoch,
                "training_losses": list(training_losses),
                "validation_losses": list(validation_losses),
                "training_supervised_losses": list(training_supervised_losses),
                "training_continuity_losses": list(training_continuity_losses),
                "validation_supervised_losses": list(validation_supervised_losses),
                "validation_continuity_losses": list(validation_continuity_losses),
                "best_validation_loss": best_validation_loss,
                "best_epoch": best_epoch,
                "learning_rate": config.learning_rate,
                "physics_loss_weight": config.physics_loss_weight,
                "continuity_residual_scale_ms": (config.continuity_residual_scale_ms),
            }
            torch.save(checkpoint, checkpoint_path)

    # Verify the saved best checkpoint can be loaded before returning.
    load_checkpoint(checkpoint_path, device="cpu")

    return TrainingV2Result(
        model_version="fno-0.5.0",
        epochs_completed=config.epochs,
        training_losses=tuple(training_losses),
        validation_losses=tuple(validation_losses),
        training_supervised_losses=tuple(training_supervised_losses),
        training_continuity_losses=tuple(training_continuity_losses),
        validation_supervised_losses=tuple(validation_supervised_losses),
        validation_continuity_losses=tuple(validation_continuity_losses),
        best_validation_loss=best_validation_loss,
        best_epoch=best_epoch,
        checkpoint_path=str(checkpoint_path),
    )
