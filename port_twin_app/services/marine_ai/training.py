"""
Training utilities for the initial marine spatial FNO.
"""

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
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")


@dataclass(frozen=True)
class TrainingResult:
    """Summary of a completed training run."""

    model_version: str
    epochs_completed: int
    training_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]
    best_validation_loss: float
    checkpoint_path: str


def _target_tensor(target: MarinePredictionResult, device: str) -> torch.Tensor:
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


def _sample_loss(
    model: MarineFNO,
    sample: MarineDatasetSample,
    device: str,
) -> torch.Tensor:
    """Calculate mean squared error for one sample."""

    inputs = request_to_tensor(sample.request, device=device)
    targets = _target_tensor(sample.target, device)

    predictions = model(inputs)

    if predictions.shape != targets.shape:
        raise ValueError(
            f"Prediction shape {tuple(predictions.shape)} does not match "
            f"target shape {tuple(targets.shape)}."
        )

    if not torch.isfinite(targets).all():
        raise ValueError(
            f"Sample {sample.sample_id!r} contains non-finite target values."
        )

    return F.mse_loss(predictions, targets)


def _evaluate(
    model: MarineFNO,
    samples: tuple[MarineDatasetSample, ...],
    device: str,
) -> float:
    """Return mean per-sample loss without updating model parameters."""

    model.eval()
    losses = []

    with torch.no_grad():
        for sample in samples:
            loss = _sample_loss(model, sample, device)
            losses.append(loss.item())

    return sum(losses) / len(losses)


def train_model(
    model: MarineFNO,
    dataset: MarineDatasetSplit,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Train an FNO and save the checkpoint with the best validation loss."""

    config = config or TrainingConfig()

    if not isinstance(model, MarineFNO):
        raise TypeError("model must be a MarineFNO instance.")
    if not isinstance(dataset, MarineDatasetSplit):
        raise TypeError("dataset must be a MarineDatasetSplit instance.")

    for partition_name in ("train", "validation", "test"):
        if not getattr(dataset, partition_name):
            raise ValueError(f"The {partition_name} partition cannot be empty.")

    device = torch.device(config.device)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    training_losses = []
    validation_losses = []
    best_validation_loss = float("inf")
    checkpoint_path = Path(config.checkpoint_path)

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses = []

        for sample in dataset.train:
            optimizer.zero_grad(set_to_none=True)
            loss = _sample_loss(model, sample, str(device))

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
        )

        training_losses.append(training_loss)
        validation_losses.append(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_checkpoint(
                model,
                checkpoint_path,
                optimizer=optimizer,
                epoch=epoch,
            )

    return TrainingResult(
        model_version=MODEL_VERSION,
        epochs_completed=config.epochs,
        training_losses=tuple(training_losses),
        validation_losses=tuple(validation_losses),
        best_validation_loss=best_validation_loss,
        checkpoint_path=str(checkpoint_path),
    )
