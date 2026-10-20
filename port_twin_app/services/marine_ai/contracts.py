"""
Input and output contracts for marine AI predictions.

This module defines the data exchanged between Django and the marine
prediction model. It does not execute inference or depend on a trained model.
"""

from dataclasses import dataclass
from typing import Literal

CONTRACT_VERSION = "1.0"


@dataclass(frozen=True)
class MarineGrid:
    """
    Defines the geographic grid used for a marine prediction.
    """

    west: float
    south: float
    east: float
    north: float
    nx: int
    ny: int
    crs: str = "EPSG:4326"

    def __post_init__(self):
        if not self.west < self.east:
            raise ValueError("west must be less than east.")

        if not self.south < self.north:
            raise ValueError("south must be less than north.")

        if not -180 <= self.west <= 180:
            raise ValueError("west must be a valid longitude.")

        if not -180 <= self.east <= 180:
            raise ValueError("east must be a valid longitude.")

        if not -90 <= self.south <= 90:
            raise ValueError("south must be a valid latitude.")

        if not -90 <= self.north <= 90:
            raise ValueError("north must be a valid latitude.")

        if self.nx < 2 or self.ny < 2:
            raise ValueError("Grid dimensions must each be at least 2.")

        if self.crs != "EPSG:4326":
            raise ValueError("The initial contract requires EPSG:4326.")


@dataclass(frozen=True)
class MarineForecastConfig:
    """
    Defines the forecast period and temporal resolution.
    """

    start_time: str
    steps: int
    step_seconds: int

    def __post_init__(self):
        from datetime import datetime

        try:
            start = datetime.fromisoformat(self.start_time.replace("Z", "+00:00"))  # noqa: FURB162
        except ValueError as exc:
            raise ValueError("start_time must be an ISO 8601 timestamp.") from exc

        if start.tzinfo is None or start.utcoffset() is None:
            raise ValueError("start_time must include a timezone.")

        if self.steps < 1:
            raise ValueError("steps must be at least 1.")

        if self.step_seconds < 1:
            raise ValueError("step_seconds must be positive.")


@dataclass(frozen=True)
class MarinePredictionRequest:
    """
    Input contract for a marine prediction.

    Spatial fields use [y, x].
    Time-varying fields use [time, y, x].
    """

    grid: MarineGrid
    forecast: MarineForecastConfig
    bathymetry_m: list[list[float]]
    initial_water_level_m: list[list[float]]
    wind_u_ms: list[list[list[float]]]
    wind_v_ms: list[list[list[float]]]
    boundary_conditions: dict[str, object]

    def __post_init__(self):
        ny, nx = self.grid.ny, self.grid.nx
        steps = self.forecast.steps

        self._validate_grid_field(self.bathymetry_m, ny, nx, "bathymetry_m")
        self._validate_grid_field(
            self.initial_water_level_m, ny, nx, "initial_water_level_m"
        )
        self._validate_time_field(self.wind_u_ms, steps, ny, nx, "wind_u_ms")
        self._validate_time_field(self.wind_v_ms, steps, ny, nx, "wind_v_ms")

        if not isinstance(self.boundary_conditions, dict):
            raise ValueError("boundary_conditions must be an object.")  # noqa: TRY004

    @staticmethod
    def _validate_grid_field(field, ny, nx, name):
        if len(field) != ny or any(len(row) != nx for row in field):
            raise ValueError(f"{name} must have shape [{ny}, {nx}].")

    @staticmethod
    def _validate_time_field(field, steps, ny, nx, name):
        if len(field) != steps or any(
            len(frame) != ny or any(len(row) != nx for row in frame) for frame in field
        ):
            raise ValueError(f"{name} must have shape [{steps}, {ny}, {nx}].")


@dataclass(frozen=True)
class MarinePredictionResult:
    """
    Output contract returned by the marine prediction model.

    All predicted fields use [time, y, x].
    """

    water_level_m: list[list[list[float]]]
    current_u_ms: list[list[list[float]]]
    current_v_ms: list[list[list[float]]]
    model_version: str
    status: Literal["success"] = "success"
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self):
        if not self.model_version.strip():
            raise ValueError("model_version must not be empty.")

        if self.status != "success":
            raise ValueError("Successful results must have status='success'.")

        if self.contract_version != CONTRACT_VERSION:
            raise ValueError("Unsupported contract version.")

        shapes = {
            self._shape(self.water_level_m),
            self._shape(self.current_u_ms),
            self._shape(self.current_v_ms),
        }

        if len(shapes) != 1:
            raise ValueError("All prediction fields must have the same shape.")

    @staticmethod
    def _shape(field):
        steps = len(field)

        if steps == 0:
            raise ValueError("Prediction fields cannot be empty.")

        ny = len(field[0])

        if ny == 0:
            raise ValueError("Prediction fields cannot have empty grids.")

        nx = len(field[0][0])

        if nx == 0:
            raise ValueError("Prediction fields cannot have empty rows.")

        if any(
            len(frame) != ny or any(len(row) != nx for row in frame) for frame in field
        ):
            raise ValueError("Prediction fields must be rectangular.")

        return steps, ny, nx
