"""Fourier Neural Operator for marine spatial forecasting."""

from pathlib import Path

import torch
from torch import nn

from .contracts import (
    CONTRACT_VERSION,
    FORECAST_STEPS,
    MarinePredictionRequest,
    MarinePredictionResult,
)

MODEL_VERSION = "fno-0.3.0"

INPUT_CHANNELS = 6
OUTPUT_CHANNELS = 3

INPUT_STATISTICS_SHAPE = (1, INPUT_CHANNELS, 1, 1)
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
    """Spatial FNO that predicts the complete fixed forecast horizon."""

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
        self.forecast_steps = FORECAST_STEPS

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

        if channels != INPUT_CHANNELS:
            raise ValueError(
                f"Expected {INPUT_CHANNELS} input channels, received {channels}."
            )

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


def request_to_tensor(
    request: MarinePredictionRequest,
    device: str = "cpu",
) -> torch.Tensor:
    """Encode a request as [1, 6, ny, nx].

    Input channels:
        0: bathymetry, positive downward in metres
        1: initial water level in metres
        2: initial eastward current velocity in m/s
        3: initial northward current velocity in m/s
        4: normalized longitude
        5: normalized latitude
    """

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

    initial_current_u = grid_tensor(
        request.initial_current_u_ms,
        "initial_current_u_ms",
    )

    initial_current_v = grid_tensor(
        request.initial_current_v_ms,
        "initial_current_v_ms",
    )

    # Normalize geographic coordinates to approximately [-1, 1].
    longitude = (
        torch.linspace(
            request.grid.west / 180.0,
            request.grid.east / 180.0,
            nx,
            dtype=torch.float32,
            device=device,
        )
        .view(1, nx)
        .expand(ny, nx)
    )

    latitude = (
        torch.linspace(
            request.grid.south / 90.0,
            request.grid.north / 90.0,
            ny,
            dtype=torch.float32,
            device=device,
        )
        .view(ny, 1)
        .expand(ny, nx)
    )

    inputs = torch.stack(
        [
            bathymetry,
            initial_water_level,
            initial_current_u,
            initial_current_v,
            longitude,
            latitude,
        ],
        dim=0,
    )

    if not torch.isfinite(inputs).all():
        raise ValueError("Encoded model inputs contain non-finite values.")

    return inputs.unsqueeze(0)


def _validate_input_statistics(
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and prepare input normalization statistics."""

    mean = torch.as_tensor(
        input_mean,
        dtype=torch.float32,
        device=device,
    )
    std = torch.as_tensor(
        input_std,
        dtype=torch.float32,
        device=device,
    )

    if tuple(mean.shape) != INPUT_STATISTICS_SHAPE:
        raise ValueError(f"input_mean must have shape {INPUT_STATISTICS_SHAPE}.")

    if tuple(std.shape) != INPUT_STATISTICS_SHAPE:
        raise ValueError(f"input_std must have shape {INPUT_STATISTICS_SHAPE}.")

    if not torch.isfinite(mean).all():
        raise ValueError("input_mean contains non-finite values.")

    if not torch.isfinite(std).all() or torch.any(std <= 0):
        raise ValueError("input_std must contain finite positive values.")

    return mean, std


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
    input_mean: torch.Tensor | None = None,
    input_std: torch.Tensor | None = None,
) -> MarinePredictionResult:
    """Run inference and optionally reverse target normalization."""

    if (target_mean is None) != (target_std is None):
        raise ValueError(
            "target_mean and target_std must either both be supplied "
            "or both be omitted."
        )
    
    if (input_mean is None) != (input_std is None):
        raise ValueError(
            "input_mean and input_std must either both be supplied "
            "or both be omitted."
        )

    model = model.to(device)
    model.eval()

    inputs = request_to_tensor(request, device=device)

    if input_mean is not None and input_std is not None:
        mean, std = _validate_input_statistics(
            input_mean,
            input_std,
            device,
        )
        inputs = (inputs - mean) / std

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

    # [1, 6, 3, ny, nx] -> three [6, ny, nx] lists.
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
    input_mean: torch.Tensor | None = None,
    input_std: torch.Tensor | None = None,
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
            "forecast_steps": model.forecast_steps,
        },
        "model_state_dict": model.state_dict(),
    }
    
    if (input_mean is None) != (input_std is None):
        raise ValueError(
            "input_mean and input_std must either both be supplied "
            "or both be omitted."
        )

    if input_mean is not None and input_std is not None:
        mean, std = _validate_input_statistics(
            input_mean,
            input_std,
            "cpu",
        )
        checkpoint["input_mean"] = mean.detach().cpu()
        checkpoint["input_std"] = std.detach().cpu()

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
    
    input_mean = checkpoint.get("input_mean")
    input_std = checkpoint.get("input_std")

    if (input_mean is None) != (input_std is None):
        raise ValueError(
            "Checkpoint must contain both input_mean and input_std."
        )

    if input_mean is not None:
        _validate_input_statistics(input_mean, input_std, device)

    architecture = checkpoint.get("architecture")

    if not isinstance(architecture, dict):
        raise ValueError("Checkpoint is missing model architecture metadata.")  # noqa: TRY004

    if architecture.get("forecast_steps") != FORECAST_STEPS:
        raise ValueError(
            "Checkpoint forecast horizon does not match the current model."
        )

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
    """Load a trained checkpoint and predict in physical units."""

    model, checkpoint = load_checkpoint(
        checkpoint_path,
        device=device,
    )

    target_mean = checkpoint.get("target_mean")
    target_std = checkpoint.get("target_std")
    input_mean = checkpoint.get("input_mean")
    input_std = checkpoint.get("input_std")

    if (target_mean is None) != (target_std is None):
        raise ValueError("Checkpoint must contain both target_mean and target_std.")

    return predict(
        request,
        model,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
        input_mean=input_mean,
        input_std=input_std,
    )
