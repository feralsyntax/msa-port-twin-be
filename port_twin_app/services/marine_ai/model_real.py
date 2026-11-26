"""Daily real-data Fourier Neural Operator."""

from pathlib import Path

import torch
from torch import nn

from .model import FNOBlock

MODEL_VERSION = "fno-real-0.1.0"
DATASET_VERSION = "copernicus-reanalysis-2020-2025"
INPUT_CONTRACT_VERSION = "marine-fno-real-1.0"

INPUT_CHANNELS = 6
OUTPUT_CHANNELS = 3
FORECAST_STEPS = 6

INPUT_CHANNEL_ORDER = (
    "initial_zos",
    "initial_uo",
    "initial_vo",
    "ocean_mask",
    "normalized_longitude",
    "normalized_latitude",
)

OUTPUT_CHANNEL_ORDER = ("zos", "uo", "vo")

INPUT_STATISTICS_SHAPE = (1, INPUT_CHANNELS, 1, 1)
TARGET_STATISTICS_SHAPE = (1, 1, OUTPUT_CHANNELS, 1, 1)


def _validate_statistics(mean, std, expected_shape, name, device="cpu"):
    """Validate normalization statistics."""

    mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
    std = torch.as_tensor(std, dtype=torch.float32, device=device)

    if tuple(mean.shape) != expected_shape:
        raise ValueError(f"{name}_mean must have shape {expected_shape}.")

    if tuple(std.shape) != expected_shape:
        raise ValueError(f"{name}_std must have shape {expected_shape}.")

    if not torch.isfinite(mean).all():
        raise ValueError(f"{name}_mean contains non-finite values.")

    if not torch.isfinite(std).all() or torch.any(std <= 0):
        raise ValueError(f"{name}_std must contain finite positive values.")

    return mean, std


class MarineFNOReal(nn.Module):
    """FNO for six-day forecasts from daily Copernicus reanalysis."""

    def __init__(
        self,
        width: int = 24,
        modes_x: int = 8,
        modes_y: int = 8,
        layers: int = 4,
    ):
        super().__init__()

        if isinstance(width, bool) or not isinstance(width, int) or width < 1:
            raise ValueError("width must be a positive integer.")

        if isinstance(layers, bool) or not isinstance(layers, int) or layers < 1:
            raise ValueError("layers must be a positive integer.")

        if (
            isinstance(modes_x, bool)
            or not isinstance(modes_x, int)
            or modes_x < 1
            or isinstance(modes_y, bool)
            or not isinstance(modes_y, int)
            or modes_y < 1
        ):
            raise ValueError("Fourier mode counts must be positive integers.")

        self.width = width
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.layers = layers
        self.forecast_steps = FORECAST_STEPS
        self.input_channels = INPUT_CHANNELS
        self.output_channels = OUTPUT_CHANNELS

        self.input_projection = nn.Conv2d(
            INPUT_CHANNELS,
            width,
            kernel_size=1,
        )

        self.blocks = nn.ModuleList(
            [FNOBlock(width, modes_x, modes_y) for _ in range(layers)]
        )

        self.output_projection = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(
                width * 2,
                FORECAST_STEPS * OUTPUT_CHANNELS,
                kernel_size=1,
            ),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return predictions shaped [batch, 6, 3, ny, nx]."""

        if inputs.ndim != 4:
            raise ValueError("Inputs must have shape [batch, channels, ny, nx].")

        batch_size, channels, ny, nx = inputs.shape

        if batch_size < 1 or ny < 1 or nx < 1:
            raise ValueError("Batch and spatial dimensions must be positive.")

        if channels != INPUT_CHANNELS:
            raise ValueError(
                f"Expected {INPUT_CHANNELS} input channels, received {channels}."
            )

        if not torch.is_floating_point(inputs):
            raise ValueError("Model inputs must have a floating-point dtype.")

        if not torch.isfinite(inputs).all():
            raise ValueError("Model inputs contain non-finite values.")

        features = self.input_projection(inputs)

        for block in self.blocks:
            features = block(features)

        outputs = self.output_projection(features)

        return outputs.reshape(
            batch_size,
            FORECAST_STEPS,
            OUTPUT_CHANNELS,
            ny,
            nx,
        )


def save_checkpoint(
    model: MarineFNOReal,
    path: str | Path,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    input_mean: torch.Tensor | None = None,
    input_std: torch.Tensor | None = None,
    target_mean: torch.Tensor | None = None,
    target_std: torch.Tensor | None = None,
) -> None:
    """Save a real-data model checkpoint and its input contract."""

    if not isinstance(model, MarineFNOReal):
        raise TypeError("Expected a MarineFNOReal model.")

    if (input_mean is None) != (input_std is None):
        raise ValueError("Both input normalization statistics must be supplied.")

    if (target_mean is None) != (target_std is None):
        raise ValueError("Both target normalization statistics must be supplied.")

    checkpoint = {
        "model_version": MODEL_VERSION,
        "dataset_version": DATASET_VERSION,
        "input_contract_version": INPUT_CONTRACT_VERSION,
        "input_channels": INPUT_CHANNELS,
        "output_channels": OUTPUT_CHANNELS,
        "forecast_steps": FORECAST_STEPS,
        "input_channel_order": list(INPUT_CHANNEL_ORDER),
        "output_channel_order": list(OUTPUT_CHANNEL_ORDER),
        "architecture": {
            "width": model.width,
            "modes_x": model.modes_x,
            "modes_y": model.modes_y,
            "layers": model.layers,
            "forecast_steps": model.forecast_steps,
        },
        "model_state_dict": model.state_dict(),
    }

    if input_mean is not None:
        mean, std = _validate_statistics(
            input_mean,
            input_std,
            INPUT_STATISTICS_SHAPE,
            "input",
        )
        checkpoint["input_mean"] = mean.detach().cpu()
        checkpoint["input_std"] = std.detach().cpu()

    if target_mean is not None:
        mean, std = _validate_statistics(
            target_mean,
            target_std,
            TARGET_STATISTICS_SHAPE,
            "target",
        )
        checkpoint["target_mean"] = mean.detach().cpu()
        checkpoint["target_std"] = std.detach().cpu()

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if epoch is not None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer.")
        checkpoint["epoch"] = epoch

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    path: str | Path,
    device: str = "cpu",
) -> tuple[MarineFNOReal, dict]:
    """Load and validate a real-data model checkpoint."""

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid real-data checkpoint format.")

    expected_metadata = {
        "model_version": MODEL_VERSION,
        "dataset_version": DATASET_VERSION,
        "input_contract_version": INPUT_CONTRACT_VERSION,
        "input_channels": INPUT_CHANNELS,
        "output_channels": OUTPUT_CHANNELS,
        "forecast_steps": FORECAST_STEPS,
        "input_channel_order": list(INPUT_CHANNEL_ORDER),
        "output_channel_order": list(OUTPUT_CHANNEL_ORDER),
    }

    for key, expected in expected_metadata.items():
        if checkpoint.get(key) != expected:
            raise ValueError(
                f"Checkpoint {key} does not match the real-data model contract."
            )

    architecture = checkpoint.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("Checkpoint architecture metadata is missing.")

    required_architecture = ("width", "modes_x", "modes_y", "layers")
    if any(key not in architecture for key in required_architecture):
        raise ValueError("Checkpoint architecture metadata is incomplete.")

    if architecture.get("forecast_steps") != FORECAST_STEPS:
        raise ValueError("Checkpoint forecast horizon does not match.")

    input_mean = checkpoint.get("input_mean")
    input_std = checkpoint.get("input_std")
    target_mean = checkpoint.get("target_mean")
    target_std = checkpoint.get("target_std")

    if (input_mean is None) != (input_std is None):
        raise ValueError("Checkpoint must contain both input statistics.")

    if (target_mean is None) != (target_std is None):
        raise ValueError("Checkpoint must contain both target statistics.")

    if input_mean is not None:
        _validate_statistics(
            input_mean,
            input_std,
            INPUT_STATISTICS_SHAPE,
            "input",
            device,
        )

    if target_mean is not None:
        _validate_statistics(
            target_mean,
            target_std,
            TARGET_STATISTICS_SHAPE,
            "target",
            device,
        )

    model = MarineFNOReal(
        width=architecture["width"],
        modes_x=architecture["modes_x"],
        modes_y=architecture["modes_y"],
        layers=architecture["layers"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    return model, checkpoint
