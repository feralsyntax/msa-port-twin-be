"""Training utilities for the initial marine spatial FNO."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .contracts import MarinePredictionResult
from .dataset import MarineDatasetSample, MarineDatasetSplit
from .model import (
    MODEL_VERSION,
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
    checkpoint_path: str = "artifacts/marine_fno.pt"

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


def _fit_target_normalization(
    samples: tuple[MarineDatasetSample, ...],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate per-variable target statistics from training samples only."""

    if not samples:
        raise ValueError("Training samples cannot be empty.")

    targets = torch.cat(
        [_target_tensor(sample.target, device) for sample in samples],
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
) -> torch.Tensor:
    """Calculate MSE against normalized target fields."""

    inputs = request_to_tensor(sample.request, device=device)
    targets = _target_tensor(sample.target, device)

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

    Target normalization is fitted exclusively on the training partition.
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
        )

        training_losses.append(training_loss)
        validation_losses.append(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch

            # Preserve normalization statistics alongside the model.
            save_checkpoint(
                model,
                checkpoint_path,
                optimizer=optimizer,
                epoch=epoch,
            )

            # Add normalization metadata to the saved checkpoint.
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            checkpoint["target_mean"] = target_mean.detach().cpu()
            checkpoint["target_std"] = target_std.detach().cpu()
            torch.save(checkpoint, checkpoint_path)

    return TrainingResult(
        model_version=MODEL_VERSION,
        epochs_completed=config.epochs,
        training_losses=tuple(training_losses),
        validation_losses=tuple(validation_losses),
        best_validation_loss=best_validation_loss,
        best_epoch=best_epoch,
        checkpoint_path=str(checkpoint_path),
    )
