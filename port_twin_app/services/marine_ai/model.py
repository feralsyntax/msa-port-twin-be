"""Fourier Neural Operator for marine spatial forecasting."""

from pathlib import Path

import torch
from torch import nn

from .contracts import (
    CONTRACT_VERSION,
    MarinePredictionRequest,
    MarinePredictionResult,
)

MODEL_VERSION = "fno-0.1.0"

BASE_INPUT_CHANNELS = 7
BOUNDARY_INPUT_CHANNELS = 8
INPUT_CHANNELS = BASE_INPUT_CHANNELS + BOUNDARY_INPUT_CHANNELS
OUTPUT_CHANNELS = 3

TARGET_STATISTICS_SHAPE = (1, 1, OUTPUT_CHANNELS, 1, 1)


class SpectralConv2d(nn.Module):
    """Two-dimensional Fourier convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_x: int,
        modes_y: int,
    ):
        super().__init__()

        if modes_x < 1 or modes_y < 1:
            raise ValueError("Fourier mode counts must be positive.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y

        scale = 1.0 / (in_channels * out_channels)

        self.weights_positive = nn.Parameter(
            scale
            * torch.rand(
                in_channels,
                out_channels,
                modes_x,
                modes_y,
                dtype=torch.cfloat,
            )
        )

        self.weights_negative = nn.Parameter(
            scale
            * torch.rand(
                in_channels,
                out_channels,
                modes_x,
                modes_y,
                dtype=torch.cfloat,
            )
        )

    @staticmethod
    def _multiply(
        inputs: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Multiply Fourier coefficients by learned channel weights."""

        return torch.einsum("bixy,ioxy->boxy", inputs, weights)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, _, nx, ny = inputs.shape

        transformed = torch.fft.rfft2(inputs, norm="ortho")

        output = torch.zeros(
            batch_size,
            self.out_channels,
            nx,
            ny // 2 + 1,
            dtype=transformed.dtype,
            device=inputs.device,
        )

        modes_x = min(self.modes_x, nx // 2)
        modes_y = min(self.modes_y, ny // 2 + 1)

        if modes_x == 0 or modes_y == 0:
            return torch.fft.irfft2(
                output,
                s=(nx, ny),
                norm="ortho",
            )

        output[:, :, :modes_x, :modes_y] = self._multiply(
            transformed[:, :, :modes_x, :modes_y],
            self.weights_positive[:, :, :modes_x, :modes_y],
        )

        output[:, :, -modes_x:, :modes_y] = self._multiply(
            transformed[:, :, -modes_x:, :modes_y],
            self.weights_negative[:, :, :modes_x, :modes_y],
        )

        return torch.fft.irfft2(
            output,
            s=(nx, ny),
            norm="ortho",
        )


class FNOBlock(nn.Module):
    """Fourier convolution combined with a local linear transformation."""

    def __init__(
        self,
        width: int,
        modes_x: int,
        modes_y: int,
    ):
        super().__init__()

        self.spectral = SpectralConv2d(
            width,
            width,
            modes_x,
            modes_y,
        )

        self.pointwise = nn.Conv2d(
            width,
            width,
            kernel_size=1,
        )

        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.spectral(inputs) + self.pointwise(inputs))


class MarineFNO(nn.Module):
    """Spatial FNO applied independently to each forecast time step.

    The same spatial model weights are shared across forecast steps.
    The model is not autoregressive and does not explicitly evolve its
    predictions from one forecast step to the next.
    """

    def __init__(
        self,
        width: int = 24,
        modes_x: int = 8,
        modes_y: int = 8,
        layers: int = 4,
    ):
        super().__init__()

        if width < 1:
            raise ValueError("width must be positive.")

        if layers < 1:
            raise ValueError("layers must be positive.")

        if modes_x < 1 or modes_y < 1:
            raise ValueError("Fourier mode counts must be positive.")

        self.width = width
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.layers = layers

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
                OUTPUT_CHANNELS,
                kernel_size=1,
            ),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return predictions shaped [batch, steps, 3, ny, nx]."""

        if inputs.ndim != 5:
            raise ValueError("Inputs must have shape [batch, steps, channels, ny, nx].")

        batch_size, steps, channels, ny, nx = inputs.shape

        if channels != INPUT_CHANNELS:
            raise ValueError(
                f"Expected {INPUT_CHANNELS} input channels, received {channels}."
            )

        # Treat each time step as a spatial sample. Model weights are shared.
        spatial_inputs = inputs.reshape(
            batch_size * steps,
            channels,
            ny,
            nx,
        )

        features = self.input_projection(spatial_inputs)

        for block in self.blocks:
            features = block(features)

        outputs = self.output_projection(features)

        return outputs.reshape(
            batch_size,
            steps,
            OUTPUT_CHANNELS,
            ny,
            nx,
        )


def _boundary_series(
    value,
    steps: int,
    boundary_name: str,
    field_name: str,
) -> list[float]:
    """Convert a boundary value or time series into a forecast-length list."""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        values = [float(value)] * steps
    elif isinstance(value, (list, tuple)) and len(value) == steps:
        values = list(value)
    else:
        raise ValueError(
            f"Boundary {boundary_name!r} field {field_name!r} "
            f"must be a number or a sequence of length {steps}."
        )

    result = []

    for item in values:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(  # noqa: TRY004
                f"Boundary {boundary_name!r} field {field_name!r} "
                "must contain numeric values."
            )

        result.append(float(item))

    return result


def request_to_tensor(
    request: MarinePredictionRequest,
    device: str = "cpu",
) -> torch.Tensor:
    """Encode a marine request as [1, steps, 15, ny, nx].

    Base channels:
        0: bathymetry, positive downward in metres
        1: initial water level in metres
        2: eastward wind component in m/s
        3: northward wind component in m/s
        4: normalized east-west coordinate
        5: normalized north-south coordinate
        6: normalized forecast time

    Boundary channels, in order:
        7: west water level
        8: west normal velocity
        9: east water level
        10: east normal velocity
        11: south water level
        12: south normal velocity
        13: north water level
        14: north normal velocity
    """

    steps = request.forecast.steps
    nx = request.grid.nx
    ny = request.grid.ny

    def grid_tensor(values, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(
            values,
            dtype=torch.float32,
            device=device,
        )

        if tuple(tensor.shape) != (ny, nx):
            raise ValueError(
                f"{name} must have shape {(ny, nx)}, received {tuple(tensor.shape)}."
            )

        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains non-finite values.")

        return tensor

    bathymetry = grid_tensor(
        request.bathymetry_m,
        "bathymetry_m",
    )

    initial_water_level = grid_tensor(
        request.initial_water_level_m,
        "initial_water_level_m",
    )

    wind_u = torch.as_tensor(
        request.wind_u_ms,
        dtype=torch.float32,
        device=device,
    )

    wind_v = torch.as_tensor(
        request.wind_v_ms,
        dtype=torch.float32,
        device=device,
    )

    expected_wind_shape = (steps, ny, nx)

    if tuple(wind_u.shape) != expected_wind_shape:
        raise ValueError(f"wind_u_ms must have shape {expected_wind_shape}.")

    if tuple(wind_v.shape) != expected_wind_shape:
        raise ValueError(f"wind_v_ms must have shape {expected_wind_shape}.")

    if not torch.isfinite(wind_u).all():
        raise ValueError("wind_u_ms contains non-finite values.")

    if not torch.isfinite(wind_v).all():
        raise ValueError("wind_v_ms contains non-finite values.")

    # Coordinate channels use a stable normalized grid representation.
    x_coordinates = (
        torch.linspace(
            -1.0,
            1.0,
            nx,
            device=device,
        )
        .view(1, 1, nx)
        .expand(steps, ny, nx)
    )

    y_coordinates = (
        torch.linspace(
            -1.0,
            1.0,
            ny,
            device=device,
        )
        .view(1, ny, 1)
        .expand(steps, ny, nx)
    )

    time_coordinates = (
        torch.linspace(
            0.0,
            1.0,
            steps,
            device=device,
        )
        .view(steps, 1, 1)
        .expand(steps, ny, nx)
    )

    base_channels = torch.stack(
        [
            bathymetry.unsqueeze(0).expand(steps, ny, nx),
            initial_water_level.unsqueeze(0).expand(steps, ny, nx),
            wind_u,
            wind_v,
            x_coordinates,
            y_coordinates,
            time_coordinates,
        ],
        dim=1,
    )

    boundary_channels = torch.zeros(
        steps,
        BOUNDARY_INPUT_CHANNELS,
        ny,
        nx,
        dtype=torch.float32,
        device=device,
    )

    boundary_conditions = request.boundary_conditions

    boundary_definitions = (
        ("west", "water_level_m", 0, "left"),
        ("west", "normal_velocity_ms", 1, "left"),
        ("east", "water_level_m", 2, "right"),
        ("east", "normal_velocity_ms", 3, "right"),
        ("south", "water_level_m", 4, "bottom"),
        ("south", "normal_velocity_ms", 5, "bottom"),
        ("north", "water_level_m", 6, "top"),
        ("north", "normal_velocity_ms", 7, "top"),
    )

    for side, field_name, channel_index, edge in boundary_definitions:
        side_conditions = boundary_conditions.get(side, {})

        if field_name not in side_conditions:
            continue

        values = _boundary_series(
            side_conditions[field_name],
            steps,
            side,
            field_name,
        )

        values_tensor = torch.tensor(
            values,
            dtype=torch.float32,
            device=device,
        ).view(steps, 1)

        if not torch.isfinite(values_tensor).all():
            raise ValueError(
                f"Boundary {side!r} field {field_name!r} contains non-finite values."
            )

        if edge == "left":
            boundary_channels[:, channel_index, :, 0] = values_tensor
        elif edge == "right":
            boundary_channels[:, channel_index, :, -1] = values_tensor
        elif edge == "bottom":
            boundary_channels[:, channel_index, 0, :] = values_tensor
        elif edge == "top":
            boundary_channels[:, channel_index, -1, :] = values_tensor

    inputs = torch.cat(
        [base_channels, boundary_channels],
        dim=1,
    )

    if not torch.isfinite(inputs).all():
        raise ValueError("Encoded model inputs contain non-finite values.")

    return inputs.unsqueeze(0)


def _validate_target_statistics(
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and prepare target normalization statistics."""

    mean = torch.as_tensor(
        target_mean,
        dtype=torch.float32,
        device=device,
    )

    std = torch.as_tensor(
        target_std,
        dtype=torch.float32,
        device=device,
    )

    if tuple(mean.shape) != TARGET_STATISTICS_SHAPE:
        raise ValueError(f"target_mean must have shape {TARGET_STATISTICS_SHAPE}.")

    if tuple(std.shape) != TARGET_STATISTICS_SHAPE:
        raise ValueError(f"target_std must have shape {TARGET_STATISTICS_SHAPE}.")

    if not torch.isfinite(mean).all():
        raise ValueError("target_mean contains non-finite values.")

    if not torch.isfinite(std).all() or torch.any(std <= 0):
        raise ValueError("target_std must contain finite positive values.")

    return mean, std


def predict(
    request: MarinePredictionRequest,
    model: MarineFNO,
    device: str = "cpu",
    target_mean: torch.Tensor | None = None,
    target_std: torch.Tensor | None = None,
) -> MarinePredictionResult:
    """Run inference and optionally reverse target normalization.

    If target_mean and target_std are supplied, model outputs are converted
    back to physical units using predictions * target_std + target_mean.

    The three output channels are water level, eastward current, and
    northward current, respectively.
    """

    if (target_mean is None) != (target_std is None):
        raise ValueError(
            "target_mean and target_std must either both be supplied "
            "or both be omitted."
        )

    model = model.to(device)
    model.eval()

    inputs = request_to_tensor(request, device=device)

    with torch.no_grad():
        outputs = model(inputs)

    if not torch.isfinite(outputs).all():
        raise ValueError("Model produced non-finite predictions.")

    if target_mean is not None and target_std is not None:
        mean, std = _validate_target_statistics(
            target_mean,
            target_std,
            device,
        )

        outputs = outputs * std + mean

    if not torch.isfinite(outputs).all():
        raise ValueError(
            "Predictions contain non-finite values after inverse normalization."
        )

    # [1, steps, 3, ny, nx] -> three [steps, ny, nx] lists.
    output_values = outputs[0].cpu()

    result = MarinePredictionResult(
        water_level_m=output_values[:, 0].tolist(),
        current_u_ms=output_values[:, 1].tolist(),
        current_v_ms=output_values[:, 2].tolist(),
        model_version=MODEL_VERSION,
    )

    result.validate_against(request)

    return result


def save_checkpoint(
    model: MarineFNO,
    path: str | Path,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
) -> None:
    """Save model architecture, weights, and optional training state."""

    checkpoint = {
        "model_version": MODEL_VERSION,
        "contract_version": CONTRACT_VERSION,
        "architecture": {
            "width": model.width,
            "modes_x": model.modes_x,
            "modes_y": model.modes_y,
            "layers": model.layers,
        },
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if epoch is not None:
        checkpoint["epoch"] = epoch

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    path: str | Path,
    device: str = "cpu",
) -> tuple[MarineFNO, dict]:
    """Load a saved FNO checkpoint and validate its contract version."""

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    if checkpoint.get("contract_version") != CONTRACT_VERSION:
        raise ValueError(
            "Checkpoint contract version does not match the current contract."
        )

    architecture = checkpoint.get("architecture")

    if not isinstance(architecture, dict):
        raise ValueError("Checkpoint is missing model architecture metadata.")  # noqa: TRY004

    model = MarineFNO(
        width=architecture["width"],
        modes_x=architecture["modes_x"],
        modes_y=architecture["modes_y"],
        layers=architecture["layers"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def predict_from_checkpoint(
    request: MarinePredictionRequest,
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> MarinePredictionResult:
    """Load a trained checkpoint and predict in physical units.

    New training checkpoints contain target normalization statistics.
    Older checkpoints without these statistics remain usable, but their
    outputs are returned without inverse normalization.
    """

    model, checkpoint = load_checkpoint(
        checkpoint_path,
        device=device,
    )

    target_mean = checkpoint.get("target_mean")
    target_std = checkpoint.get("target_std")

    if (target_mean is None) != (target_std is None):
        raise ValueError("Checkpoint must contain both target_mean and target_std.")

    return predict(
        request,
        model,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
    )
