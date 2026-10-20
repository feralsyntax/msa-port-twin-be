"""
Fourier Neural Operator for marine spatial-field forecasting.
"""

from pathlib import Path

import torch
from torch import nn

from .contracts import (
    CONTRACT_VERSION,
    MarinePredictionRequest,
    MarinePredictionResult,
)

MODEL_VERSION = "fno-0.1.0"

# Channels: bathymetry, initial water level, two wind components,
# longitude, latitude, forecast time, and two boundary variables
# (water level and normal velocity) for each of four edges.
BASE_INPUT_CHANNELS = 7
BOUNDARY_INPUT_CHANNELS = 8
INPUT_CHANNELS = BASE_INPUT_CHANNELS + BOUNDARY_INPUT_CHANNELS
OUTPUT_CHANNELS = 3


class SpectralConv2d(nn.Module):
    """
    Learnable Fourier convolution over the two spatial dimensions.
    """

    def __init__(self, in_channels, out_channels, modes_x, modes_y):
        super().__init__()

        if modes_x < 1 or modes_y < 1:
            raise ValueError("Fourier mode counts must be positive.")

        self.modes_x = modes_x
        self.modes_y = modes_y
        scale = 1 / (in_channels * out_channels) ** 0.5

        self.weights_positive = nn.Parameter(
            scale
            * torch.randn(
                in_channels,
                out_channels,
                modes_x,
                modes_y,
                dtype=torch.cfloat,
            )
        )
        self.weights_negative = nn.Parameter(
            scale
            * torch.randn(
                in_channels,
                out_channels,
                modes_x,
                modes_y,
                dtype=torch.cfloat,
            )
        )

    @staticmethod
    def _multiply_modes(inputs, weights):
        return torch.einsum("bixy,ioxy->boxy", inputs, weights)

    def forward(self, inputs):
        batch, _, height, width = inputs.shape
        transformed = torch.fft.rfft2(inputs, norm="ortho")

        # Leave room for the positive and negative frequency bands.
        mx = min(self.modes_x, height // 2)
        my = min(self.modes_y, width // 2 + 1)

        output = torch.zeros(
            batch,
            self.weights_positive.shape[1],
            height,
            width // 2 + 1,
            dtype=transformed.dtype,
            device=inputs.device,
        )

        if mx and my:
            output[:, :, :mx, :my] = self._multiply_modes(
                transformed[:, :, :mx, :my],
                self.weights_positive[:, :, :mx, :my],
            )
            output[:, :, -mx:, :my] = self._multiply_modes(
                transformed[:, :, -mx:, :my],
                self.weights_negative[:, :, :mx, :my],
            )

        return torch.fft.irfft2(
            output,
            s=(height, width),
            norm="ortho",
        )


class FNOBlock(nn.Module):
    """
    Spectral mixing combined with a local channel projection.
    """

    def __init__(self, width, modes_x, modes_y):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes_x, modes_y)
        self.local = nn.Conv2d(width, width, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, inputs):
        return self.activation(self.spectral(inputs) + self.local(inputs))


class MarineFNO(nn.Module):
    """
    Map marine conditions to forecast water levels and current fields.

    Tensor input:
        [batch, forecast_steps, input_channels, ny, nx]

    Tensor output:
        [batch, forecast_steps, 3, ny, nx]

    Output channels are water level, eastward current, and northward current.
    """

    def __init__(
        self,
        width=24,
        modes_x=8,
        modes_y=8,
        layers=4,
    ):
        super().__init__()

        if width < 1 or layers < 1:
            raise ValueError("width and layers must be positive.")

        self.width = width
        self.input_projection = nn.Conv2d(INPUT_CHANNELS, width, kernel_size=1)
        self.blocks = nn.ModuleList(
            FNOBlock(width, modes_x, modes_y) for _ in range(layers)
        )
        self.output_projection = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, OUTPUT_CHANNELS, kernel_size=1),
        )

    def forward(self, inputs):
        if inputs.ndim != 5:
            raise ValueError("inputs must have shape [batch, steps, channels, ny, nx].")

        batch, steps, channels, ny, nx = inputs.shape
        if channels != INPUT_CHANNELS:
            raise ValueError(
                f"Expected {INPUT_CHANNELS} input channels, got {channels}."
            )

        # Apply the same spatial operator at every forecast step.
        inputs = inputs.reshape(batch * steps, channels, ny, nx)
        hidden = self.input_projection(inputs)

        for block in self.blocks:
            hidden = block(hidden) + hidden

        predictions = self.output_projection(hidden)
        return predictions.reshape(batch, steps, OUTPUT_CHANNELS, ny, nx)


def request_to_tensor(request, device="cpu"):
    """
    Encode a validated request as [1, steps, channels, ny, nx].
    """

    if not isinstance(request, MarinePredictionRequest):
        raise TypeError("request must be a MarinePredictionRequest.")

    grid = request.grid
    forecast = request.forecast
    ny, nx = grid.ny, grid.nx
    steps = forecast.steps

    def field(values):
        return torch.as_tensor(values, dtype=torch.float32, device=device)

    bathymetry = field(request.bathymetry_m)[None, None]
    initial_level = field(request.initial_water_level_m)[None, None]
    wind_u = field(request.wind_u_ms)[None, :, None]
    wind_v = field(request.wind_v_ms)[None, :, None]

    # Normalized geographic coordinates preserve spatial location.
    longitude = (
        torch.linspace(0, 1, nx, device=device)
        .view(1, 1, 1, 1, nx)
        .expand(1, steps, 1, ny, nx)
    )
    latitude = (
        torch.linspace(0, 1, ny, device=device)
        .view(1, 1, 1, ny, 1)
        .expand(1, steps, 1, ny, nx)
    )
    time = (
        torch.linspace(0, 1, steps, device=device)
        .view(1, steps, 1, 1, 1)
        .expand(1, steps, 1, ny, nx)
    )

    static_fields = torch.cat(
        [
            bathymetry.expand(1, steps, 1, ny, nx),
            initial_level.expand(1, steps, 1, ny, nx),
            wind_u,
            wind_v,
            longitude,
            latitude,
            time,
        ],
        dim=2,
    )

    # Boundary channels are ordered west, east, south, north.
    # For each edge, encode water level first and normal velocity second.
    boundary_fields = torch.zeros(
        1,
        steps,
        BOUNDARY_INPUT_CHANNELS,
        ny,
        nx,
        dtype=torch.float32,
        device=device,
    )

    edges = ("west", "east", "south", "north")
    for edge_index, edge in enumerate(edges):
        condition = request.boundary_conditions.get(edge, {})
        for variable_index, variable in enumerate(
            ("water_level_m", "normal_velocity_ms")
        ):
            if variable not in condition:
                continue

            values = field(condition[variable])
            channel = edge_index * 2 + variable_index

            if edge == "west":
                boundary_fields[0, :, channel, :, 0] = values[:, None]
            elif edge == "east":
                boundary_fields[0, :, channel, :, -1] = values[:, None]
            elif edge == "south":
                boundary_fields[0, :, channel, 0, :] = values[:, None]
            else:
                boundary_fields[0, :, channel, -1, :] = values[:, None]

    return torch.cat([static_fields, boundary_fields], dim=2)


def predict(request, model, device="cpu"):
    """
    Run inference and return the existing public prediction contract.
    """

    model = model.to(device)
    model.eval()

    inputs = request_to_tensor(request, device=device)

    with torch.no_grad():
        output = model(inputs)[0]

    output = output.detach().cpu()
    result = MarinePredictionResult(
        water_level_m=output[:, 0].tolist(),
        current_u_ms=output[:, 1].tolist(),
        current_v_ms=output[:, 2].tolist(),
        model_version=MODEL_VERSION,
        contract_version=CONTRACT_VERSION,
    )
    result.validate_against(request)
    return result


def save_checkpoint(model, path, optimizer=None, epoch=None):
    """
    Save model weights and enough architecture metadata to rebuild it.
    """

    checkpoint = {
        "model_version": MODEL_VERSION,
        "contract_version": CONTRACT_VERSION,
        "width": model.width,
        "modes_x": model.blocks[0].spectral.modes_x,
        "modes_y": model.blocks[0].spectral.modes_y,
        "layers": len(model.blocks),
        "state_dict": model.state_dict(),
        "epoch": epoch,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path, device="cpu"):
    """
    Restore an FNO checkpoint and return the model and checkpoint data.
    """

    checkpoint = torch.load(path, map_location=device, weights_only=True)

    if checkpoint.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("Checkpoint contract version does not match.")

    model = MarineFNO(
        width=checkpoint["width"],
        modes_x=checkpoint["modes_x"],
        modes_y=checkpoint["modes_y"],
        layers=checkpoint["layers"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return model, checkpoint
