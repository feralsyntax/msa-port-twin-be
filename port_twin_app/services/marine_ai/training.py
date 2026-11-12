"""Training utilities for the initial marine spatial FNO."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .contracts import MarinePredictionResult
from .dataset import MarineDatasetSample, MarineDatasetSplit
from .model import (
    MODEL_VERSION,
    PREDICTION_MODE_ABSOLUTE,
    PREDICTION_MODE_RESIDUAL,
    RESIDUAL_MODEL_VERSION,
    SUPPORTED_PREDICTION_MODES,
    MarineFNO,
    request_to_tensor,
    save_checkpoint,
)


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for supervised FNO training."""

    epochs: int = 20
    learning_rate: float = 1e-3
    device: str = "cpu"
    checkpoint_path: str = "artifacts/marine_fno_normalized.pt"
    seed: int = 42
    prediction_mode: str = PREDICTION_MODE_ABSOLUTE

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

        if self.prediction_mode not in SUPPORTED_PREDICTION_MODES:
            raise ValueError(f"Unsupported prediction mode: {self.prediction_mode!r}.")


@dataclass(frozen=True)
class TrainingResult:
    """Summary of a completed training run."""

    model_version: str
    epochs_completed: int
    training_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]
    best_validation_loss: float
    best_epoch: int
    checkpoint_path: str


def _target_tensor(
    target: MarinePredictionResult,
    device: str,
) -> torch.Tensor:
    """Stack target fields as [1, steps, 3, ny, nx]."""

    fields = torch.tensor(
        [
            target.water_level_m,
            target.current_u_ms,
            target.current_v_ms,
        ],
        dtype=torch.float32,
        device=device,
    )

    # [3, steps, ny, nx] -> [1, steps, 3, ny, nx]
    return fields.permute(1, 0, 2, 3).unsqueeze(0)


def _training_target_tensor(
    sample: MarineDatasetSample,
    device: str,
    prediction_mode: str = PREDICTION_MODE_ABSOLUTE,
) -> torch.Tensor:
    """Return either absolute future fields or changes from initial fields."""

    if prediction_mode not in SUPPORTED_PREDICTION_MODES:
        raise ValueError(f"Unsupported prediction mode: {prediction_mode!r}.")

    targets = _target_tensor(sample.target, device)

    if prediction_mode == PREDICTION_MODE_ABSOLUTE:
        return targets

    # request_to_tensor channels 1:4 are the initial physical fields:
    # water level, eastward current, and northward current.
    initial_state = request_to_tensor(
        sample.request,
        device=device,
    )[:, 1:4].unsqueeze(1)

    return targets - initial_state


def _fit_input_normalization(
    samples: tuple[MarineDatasetSample, ...],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate per-channel input statistics from training samples only."""

    if not samples:
        raise ValueError("Training samples cannot be empty.")

    inputs = torch.cat(
        [request_to_tensor(sample.request, device=device) for sample in samples],
        dim=0,
    )

    # Calculate one mean and standard deviation per input channel.
    # Result shape: [1, 6, 1, 1].
    mean = inputs.mean(dim=(0, 2, 3), keepdim=True)
    std = inputs.std(dim=(0, 2, 3), keepdim=True, unbiased=False)

    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("Training inputs contain non-finite values.")

    # Constant channels do not need scaling.
    std = torch.where(std < 1e-6, torch.ones_like(std), std)

    return mean, std


def _fit_target_normalization(
    samples: tuple[MarineDatasetSample, ...],
    device: str,
    prediction_mode: str = PREDICTION_MODE_ABSOLUTE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate per-variable target statistics from training samples only.

    In residual mode, statistics are fitted to future-minus-initial changes,
    rather than to absolute future field values.
    """

    if not samples:
        raise ValueError("Training samples cannot be empty.")

    if prediction_mode not in SUPPORTED_PREDICTION_MODES:
        raise ValueError(f"Unsupported prediction mode: {prediction_mode!r}.")

    targets = torch.cat(
        [
            _training_target_tensor(sample, device, prediction_mode)
            for sample in samples
        ],
        dim=0,
    )

    # Calculate one mean and standard deviation per output variable.
    # Result shape: [1, 1, 3, 1, 1].
    mean = targets.mean(dim=(0, 1, 3, 4), keepdim=True)
    std = targets.std(dim=(0, 1, 3, 4), keepdim=True, unbiased=False)

    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("Training targets contain non-finite values.")

    # Avoid division by zero for constant-valued output channels.
    std = torch.where(std < 1e-6, torch.ones_like(std), std)

    return mean, std


def _normalized_sample_loss(
    model: MarineFNO,
    sample: MarineDatasetSample,
    device: str,
    mean: torch.Tensor,
    std: torch.Tensor,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    prediction_mode: str = PREDICTION_MODE_ABSOLUTE,
) -> torch.Tensor:
    """Calculate MSE against normalized absolute or residual target fields."""

    if prediction_mode not in SUPPORTED_PREDICTION_MODES:
        raise ValueError(f"Unsupported prediction mode: {prediction_mode!r}.")

    inputs = request_to_tensor(sample.request, device=device)
    inputs = (inputs - input_mean) / input_std

    targets = _training_target_tensor(
        sample,
        device,
        prediction_mode,
    )

    if not torch.isfinite(targets).all():
        raise ValueError(
            f"Sample {sample.sample_id!r} contains non-finite target values."
        )

    normalized_targets = (targets - mean) / std
    predictions = model(inputs)

    if predictions.shape != normalized_targets.shape:
        raise ValueError(
            f"Prediction shape {tuple(predictions.shape)} does not match "
            f"target shape {tuple(normalized_targets.shape)}."
        )

    if not torch.isfinite(predictions).all():
        raise ValueError(
            f"Model produced non-finite predictions for sample {sample.sample_id!r}."
        )

    return F.mse_loss(predictions, normalized_targets)


def _evaluate(
    model: MarineFNO,
    samples: tuple[MarineDatasetSample, ...],
    device: str,
    mean: torch.Tensor,
    std: torch.Tensor,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    prediction_mode: str = PREDICTION_MODE_ABSOLUTE,
) -> float:
    """Return mean normalized loss without updating model parameters."""

    if not samples:
        raise ValueError("Evaluation samples cannot be empty.")

    model.eval()
    losses = []

    with torch.no_grad():
        for sample in samples:
            loss = _normalized_sample_loss(
                model,
                sample,
                device,
                mean,
                std,
                input_mean,
                input_std,
                prediction_mode,
            )

            if not torch.isfinite(loss):
                raise ValueError(
                    f"Non-finite evaluation loss for sample {sample.sample_id!r}."
                )

            losses.append(loss.item())

    return sum(losses) / len(losses)


def train_model(
    model: MarineFNO,
    dataset: MarineDatasetSplit,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Train an FNO and save the checkpoint with the best validation loss.

    Input and target normalization are fitted exclusively on the training
    partition. In residual mode, targets represent future-minus-initial
    field changes.

    Validation is used for checkpoint selection; the test partition is
    intentionally not used during training.
    """

    config = config or TrainingConfig()

    if not isinstance(model, MarineFNO):
        raise TypeError("model must be a MarineFNO instance.")

    if not isinstance(dataset, MarineDatasetSplit):
        raise TypeError("dataset must be a MarineDatasetSplit instance.")

    for partition_name in ("train", "validation", "test"):
        if not getattr(dataset, partition_name):
            raise ValueError(f"The {partition_name} partition cannot be empty.")

    device = torch.device(config.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")

    model = model.to(device)

    target_mean, target_std = _fit_target_normalization(
        dataset.train,
        str(device),
        config.prediction_mode,
    )

    input_mean, input_std = _fit_input_normalization(
        dataset.train,
        str(device),
    )

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

        for sample in dataset.train:
            optimizer.zero_grad(set_to_none=True)

            loss = _normalized_sample_loss(
                model,
                sample,
                str(device),
                target_mean,
                target_std,
                input_mean,
                input_std,
                config.prediction_mode,
            )

            if not torch.isfinite(loss):
                raise ValueError(
                    f"Non-finite training loss for sample {sample.sample_id!r}."
                )

            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        training_loss = sum(epoch_losses) / len(epoch_losses)

        validation_loss = _evaluate(
            model,
            dataset.validation,
            str(device),
            target_mean,
            target_std,
            input_mean,
            input_std,
            config.prediction_mode,
        )

        training_losses.append(training_loss)
        validation_losses.append(validation_loss)

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
                prediction_mode=config.prediction_mode,
            )

            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            checkpoint["target_mean"] = target_mean.detach().cpu()
            checkpoint["target_std"] = target_std.detach().cpu()

            checkpoint["training"] = {
                "seed": config.seed,
                "epochs_completed": epoch,
                "training_losses": list(training_losses),
                "validation_losses": list(validation_losses),
                "best_validation_loss": best_validation_loss,
                "best_epoch": best_epoch,
                "learning_rate": config.learning_rate,
            }

            torch.save(checkpoint, checkpoint_path)

    model_version = (
        RESIDUAL_MODEL_VERSION
        if config.prediction_mode == PREDICTION_MODE_RESIDUAL
        else MODEL_VERSION
    )

    return TrainingResult(
        model_version=model_version,
        epochs_completed=config.epochs,
        training_losses=tuple(training_losses),
        validation_losses=tuple(validation_losses),
        best_validation_loss=best_validation_loss,
        best_epoch=best_epoch,
        checkpoint_path=str(checkpoint_path),
    )
