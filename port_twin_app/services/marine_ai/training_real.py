"""Training utilities for the daily real-data marine FNO."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .model_real import (
    FORECAST_STEPS,
    INPUT_CHANNELS,
    INPUT_STATISTICS_SHAPE,
    OUTPUT_CHANNELS,
    TARGET_STATISTICS_SHAPE,
    MarineFNOReal,
    save_checkpoint,
)
from .preparation_real import (
    MarineFNORealSample,
    MarineFNORealSplit,
)


@dataclass(frozen=True)
class TrainingRealConfig:
    """Configuration for real-data FNO training."""

    epochs: int = 20
    learning_rate: float = 1e-3
    batch_size: int = 16
    device: str = "cpu"
    checkpoint_path: str = "artifacts/marine_fno_real.pt"
    seed: int = 42
    num_workers: int = 0

    def __post_init__(self):
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int):
            raise ValueError("epochs must be a positive integer.")
        if self.epochs < 1:
            raise ValueError("epochs must be a positive integer.")

        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not np.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate must be positive and finite.")

        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise ValueError("batch_size must be a positive integer.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")

        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string.")

        if (
            not isinstance(self.checkpoint_path, str)
            or not self.checkpoint_path.strip()
        ):
            raise ValueError("checkpoint_path must be a non-empty string.")

        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be a non-negative integer.")
        if self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")

        if (
            isinstance(self.num_workers, bool)
            or not isinstance(self.num_workers, int)
            or self.num_workers < 0
        ):
            raise ValueError("num_workers must be a non-negative integer.")


@dataclass(frozen=True)
class TrainingRealResult:
    """Summary of a completed real-data training run."""

    model_version: str
    epochs_completed: int
    training_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]
    best_validation_loss: float
    best_epoch: int
    checkpoint_path: str


def _sample_arrays(sample: MarineFNORealSample):
    """Validate one prepared sample."""

    inputs = np.asarray(sample.inputs, dtype=np.float32)
    target = np.asarray(sample.target, dtype=np.float32)

    if inputs.ndim != 3 or inputs.shape[0] != INPUT_CHANNELS:
        raise ValueError(f"Sample {sample.sample_id!r} has invalid input shape.")

    _, ny, nx = inputs.shape

    if target.shape != (FORECAST_STEPS, OUTPUT_CHANNELS, ny, nx):
        raise ValueError(f"Sample {sample.sample_id!r} has invalid target shape.")

    mask = inputs[3] > 0.5

    if not mask.any():
        raise ValueError(f"Sample {sample.sample_id!r} has no ocean cells.")

    if not np.isfinite(inputs).all():
        raise ValueError(f"Sample {sample.sample_id!r} has non-finite inputs.")

    if not np.isfinite(target[:, :, mask]).all():
        raise ValueError(f"Sample {sample.sample_id!r} has invalid ocean targets.")

    if np.isfinite(target[:, :, ~mask]).any():
        raise ValueError(f"Sample {sample.sample_id!r} must use NaN for land targets.")

    return inputs, target, mask


def _stack_samples(
    samples: tuple[MarineFNORealSample, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert prepared samples to batched tensors."""

    if not samples:
        raise ValueError("Cannot build tensors from an empty partition.")

    inputs_list = []
    targets_list = []
    masks_list = []

    reference_mask = None
    reference_shape = None

    for sample in samples:
        inputs, target, mask = _sample_arrays(sample)

        if reference_shape is None:
            reference_shape = inputs.shape
            reference_mask = mask
        elif inputs.shape != reference_shape:
            raise ValueError("Samples do not share the same input shape.")
        elif not np.array_equal(mask, reference_mask):
            raise ValueError("Samples do not share the same ocean mask.")

        # Land target NaNs become zero; the mask excludes them from loss.
        target = np.where(mask[None, None, :, :], target, 0.0)

        inputs_list.append(inputs)
        targets_list.append(target)
        masks_list.append(mask)

    return (
        torch.from_numpy(np.stack(inputs_list)),
        torch.from_numpy(np.stack(targets_list)),
        torch.from_numpy(np.stack(masks_list)),
    )


def _fit_input_normalization(
    inputs: torch.Tensor,
    masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit input statistics using training samples and ocean cells only."""

    values = []

    for sample_inputs, mask in zip(inputs, masks):
        values.append(sample_inputs[:, mask])

    combined = torch.cat(values, dim=1)

    mean = combined.mean(dim=1).reshape(INPUT_STATISTICS_SHAPE)
    std = combined.std(dim=1, unbiased=False).reshape(INPUT_STATISTICS_SHAPE)

    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("Training inputs contain non-finite values.")

    std = torch.where(std < 1e-6, torch.ones_like(std), std)

    return mean, std


def _fit_target_normalization(
    targets: torch.Tensor,
    masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit target statistics using training ocean cells only."""

    values = []

    for sample_targets, mask in zip(targets, masks):
        # [forecast_steps, output_channels, water_cells]
        values.append(sample_targets[:, :, mask])

    combined = torch.cat(values, dim=0)

    mean = combined.mean(dim=(0, 2)).reshape(TARGET_STATISTICS_SHAPE)
    std = combined.std(dim=(0, 2), unbiased=False).reshape(TARGET_STATISTICS_SHAPE)

    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("Training targets contain non-finite values.")

    std = torch.where(std < 1e-6, torch.ones_like(std), std)

    return mean, std


def masked_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Calculate MSE over ocean cells, excluding land."""

    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have matching shapes.")

    if predictions.ndim != 5:
        raise ValueError(
            "Predictions must have shape [batch, steps, channels, ny, nx]."
        )

    if masks.shape != (
        predictions.shape[0],
        predictions.shape[3],
        predictions.shape[4],
    ):
        raise ValueError("Masks do not match prediction dimensions.")

    if not torch.isfinite(predictions).all():
        raise ValueError("Predictions contain non-finite values.")

    if not torch.isfinite(targets).all():
        raise ValueError("Targets must be finite before loss calculation.")

    expanded_mask = masks[:, None, None, :, :].to(
        device=predictions.device,
        dtype=predictions.dtype,
    )

    squared_error = (predictions - targets).square() * expanded_mask
    denominator = expanded_mask.sum() * predictions.shape[1] * predictions.shape[2]

    if denominator <= 0:
        raise ValueError("Loss mask contains no valid ocean cells.")

    return squared_error.sum() / denominator


def _evaluate(
    model: MarineFNOReal,
    loader: DataLoader,
    device: torch.device,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> float:
    """Evaluate mean masked validation loss."""

    model.eval()
    losses = []

    with torch.no_grad():
        for inputs, targets, masks in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            normalized_inputs = (inputs - input_mean) / input_std
            normalized_targets = (targets - target_mean) / target_std

            predictions = model(normalized_inputs)

            loss = masked_mse_loss(
                predictions,
                normalized_targets,
                masks,
            )

            if not torch.isfinite(loss):
                raise ValueError("Validation produced a non-finite loss.")

            losses.append(loss.item())

    if not losses:
        raise ValueError("Validation loader is empty.")

    return float(np.mean(losses))


def train_model_real(
    model: MarineFNOReal,
    dataset: MarineFNORealSplit,
    config: TrainingRealConfig | None = None,
) -> TrainingRealResult:
    """Train the daily real-data FNO with training-only normalization."""

    config = config or TrainingRealConfig()

    if not isinstance(model, MarineFNOReal):
        raise TypeError("model must be a MarineFNOReal instance.")

    if not isinstance(dataset, MarineFNORealSplit):
        raise TypeError("dataset must be a MarineFNORealSplit instance.")

    for partition in ("train", "validation", "test"):
        if not getattr(dataset, partition):
            raise ValueError(f"The {partition} partition cannot be empty.")

    torch.manual_seed(config.seed)

    device = torch.device(config.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")

    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA training devices are supported.")

    train_inputs, train_targets, train_masks = _stack_samples(dataset.train)
    validation_inputs, validation_targets, validation_masks = _stack_samples(
        dataset.validation
    )

    # Test data is checked for consistency but never used for fitting.
    _stack_samples(dataset.test)

    if train_inputs.shape[1:] != validation_inputs.shape[1:]:
        raise ValueError("Training and validation input shapes do not match.")

    if not torch.equal(train_masks[0], validation_masks[0]):
        raise ValueError("Training and validation ocean masks do not match.")

    input_mean, input_std = _fit_input_normalization(
        train_inputs,
        train_masks,
    )
    target_mean, target_std = _fit_target_normalization(
        train_targets,
        train_masks,
    )

    input_mean = input_mean.to(device)
    input_std = input_std.to(device)
    target_mean = target_mean.to(device)
    target_std = target_std.to(device)

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    train_loader = DataLoader(
        TensorDataset(train_inputs, train_targets, train_masks),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=generator,
    )

    validation_loader = DataLoader(
        TensorDataset(validation_inputs, validation_targets, validation_masks),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    training_losses = []
    validation_losses = []
    best_validation_loss = float("inf")
    best_epoch = 0
    checkpoint_path = Path(config.checkpoint_path)

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses = []

        for inputs, targets, masks in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            normalized_inputs = (inputs - input_mean) / input_std
            normalized_targets = (targets - target_mean) / target_std

            optimizer.zero_grad(set_to_none=True)

            predictions = model(normalized_inputs)

            loss = masked_mse_loss(
                predictions,
                normalized_targets,
                masks,
            )

            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss at epoch {epoch}.")

            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        training_loss = float(np.mean(epoch_losses))

        validation_loss = _evaluate(
            model,
            validation_loader,
            device,
            input_mean,
            input_std,
            target_mean,
            target_std,
        )

        training_losses.append(training_loss)
        validation_losses.append(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch

            save_checkpoint(
                model=model,
                path=checkpoint_path,
                optimizer=optimizer,
                epoch=epoch,
                input_mean=input_mean,
                input_std=input_std,
                target_mean=target_mean,
                target_std=target_std,
            )

    return TrainingRealResult(
        model_version="fno-real-0.1.0",
        epochs_completed=config.epochs,
        training_losses=tuple(training_losses),
        validation_losses=tuple(validation_losses),
        best_validation_loss=best_validation_loss,
        best_epoch=best_epoch,
        checkpoint_path=str(checkpoint_path),
    )
