"""
Input and output contracts for marine AI predictions.

Versioned input and output contracts for marine AI predictions.

Conventions:
- Geographic bounds use EPSG:4326 (longitude, latitude).
- Spatial arrays use [y, x].
- Time-varying arrays use [time, y, x].
- Bathymetry is positive-downward water depth in metres.
- Water level is relative to a documented vertical datum.
- u is eastward velocity; v is northward velocity.
- Wind components use eastward/northward directions in m/s.
- Boundary values are supplied at each forecast step.

This module defines the data exchanged between Django and the marine
prediction model. It does not execute inference or depend on a trained model.
"""

import math
from dataclasses import dataclass
from datetime import datetime
from numbers import Real
from typing import Literal

CONTRACT_VERSION = "1.0"

BOUNDARY_EDGES = frozenset({"west", "east", "south", "north"})
BOUNDARY_VARIABLES = frozenset({"water_level_m", "normal_velocity_ms"})


def validate_finite_numbers(value, name):
    """Reject booleans, non-numeric values, NaN, and infinity."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must contain numbers, not booleans.")  # noqa: TRY004

    if isinstance(value, Real):
        if not math.isfinite(value):
            raise ValueError(f"{name} must contain only finite numbers.")
        return

    if isinstance(value, list):
        for item in value:
            validate_finite_numbers(item, name)
        return

    raise ValueError(f"{name} must contain only numeric values.")


def validate_rectangular_field(field, dimensions, name):
    """Validate a nested-list tensor against an expected shape."""
    if not isinstance(field, list):
        raise ValueError(f"{name} must be a list.")  # noqa: TRY004

    if not dimensions:
        raise ValueError("At least one dimension is required.")

    expected_length = dimensions[0]

    if len(field) != expected_length:
        raise ValueError(f"{name} must have shape {dimensions}.")

    if len(dimensions) == 1:
        validate_finite_numbers(field, name)
        return

    for item in field:
        validate_rectangular_field(item, dimensions[1:], name)


@dataclass(frozen=True)
class MarineGrid:
    """Geographic bounds and grid resolution."""

    west: float
    south: float
    east: float
    north: float
    nx: int
    ny: int
    crs: str = "EPSG:4326"

    def __post_init__(self):
        for name in ("west", "south", "east", "north"):
            value = getattr(self, name)

            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be numeric.")  # noqa: TRY004

            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")

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

        for name in ("nx", "ny"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer.")  # noqa: TRY004
            if value < 2:
                raise ValueError(f"{name} must be at least 2.")

        if self.crs != "EPSG:4326":
            raise ValueError("The initial contract requires EPSG:4326.")


@dataclass(frozen=True)
class MarineForecastConfig:
    """Forecast start time and temporal resolution."""

    start_time: str
    steps: int
    step_seconds: int

    def __post_init__(self):
        if not isinstance(self.start_time, str):
            raise ValueError("start_time must be a string.")  # noqa: TRY004

        for name, value in (
            ("steps", self.steps),
            ("step_seconds", self.step_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer.")  # noqa: TRY004

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
    Model inputs.

    Grid fields: [y, x].
    Wind fields: [time, y, x].

    boundary_conditions maps an edge to exactly one variable:
    water_level_m or normal_velocity_ms. Its values are lists
    containing one finite value per forecast step.
    """

    grid: MarineGrid
    forecast: MarineForecastConfig
    bathymetry_m: list[list[float]]
    initial_water_level_m: list[list[float]]
    wind_u_ms: list[list[list[float]]]
    wind_v_ms: list[list[list[float]]]
    boundary_conditions: dict[str, dict[str, list[float]]]

    def __post_init__(self):
        if not isinstance(self.grid, MarineGrid):
            raise ValueError("grid must be a MarineGrid.")  # noqa: TRY004

        if not isinstance(self.forecast, MarineForecastConfig):
            raise ValueError("forecast must be a MarineForecastConfig.")  # noqa: TRY004

        ny, nx = self.grid.ny, self.grid.nx
        steps = self.forecast.steps

        validate_rectangular_field(self.bathymetry_m, [ny, nx], "bathymetry_m")
        validate_rectangular_field(
            self.initial_water_level_m,
            [ny, nx],
            "initial_water_level_m",
        )
        validate_rectangular_field(self.wind_u_ms, [steps, ny, nx], "wind_u_ms")
        validate_rectangular_field(self.wind_v_ms, [steps, ny, nx], "wind_v_ms")

        if not isinstance(self.boundary_conditions, dict):
            raise ValueError("boundary_conditions must be an object.")  # noqa: TRY004

        if not self.boundary_conditions:
            raise ValueError("At least one boundary condition must be supplied.")

        unknown_edges = set(self.boundary_conditions) - BOUNDARY_EDGES
        if unknown_edges:
            raise ValueError(f"Unsupported boundary edges: {sorted(unknown_edges)}.")

        for edge, condition in self.boundary_conditions.items():
            if not isinstance(condition, dict):
                raise ValueError(f"{edge} boundary condition must be an object.")  # noqa: TRY004

            if len(condition) != 1:
                raise ValueError(f"{edge} must define exactly one boundary variable.")

            variable, values = next(iter(condition.items()))

            if variable not in BOUNDARY_VARIABLES:
                raise ValueError(f"Unsupported boundary variable: {variable}.")

            validate_rectangular_field(values, [steps], f"{edge}.{variable}")


@dataclass(frozen=True)
class MarinePredictionResult:
    """
    Successful model output.

    Predicted fields use [time, y, x].
    """

    water_level_m: list[list[list[float]]]
    current_u_ms: list[list[list[float]]]
    current_v_ms: list[list[list[float]]]
    model_version: str
    status: Literal["success"] = "success"
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self):
        if not isinstance(self.model_version, str) or not self.model_version.strip():
            raise ValueError("model_version must be a non-empty string.")

        if self.status != "success":
            raise ValueError("Successful results must have status='success'.")

        if self.contract_version != CONTRACT_VERSION:
            raise ValueError("Unsupported contract version.")

        shapes = {
            self._shape(self.water_level_m, "water_level_m"),
            self._shape(self.current_u_ms, "current_u_ms"),
            self._shape(self.current_v_ms, "current_v_ms"),
        }

        if len(shapes) != 1:
            raise ValueError("All prediction fields must have the same shape.")

    def validate_against(self, request: MarinePredictionRequest):
        """Ensure every output matches the original request dimensions."""
        if not isinstance(request, MarinePredictionRequest):
            raise ValueError("request must be a MarinePredictionRequest.")  # noqa: TRY004

        expected = (
            request.forecast.steps,
            request.grid.ny,
            request.grid.nx,
        )

        for name, field in (
            ("water_level_m", self.water_level_m),
            ("current_u_ms", self.current_u_ms),
            ("current_v_ms", self.current_v_ms),
        ):
            actual = self._shape(field, name)
            if actual != expected:
                raise ValueError(f"{name} has shape {actual}; expected {expected}.")

    @staticmethod
    def _shape(field, name):
        if not isinstance(field, list) or not field:
            raise ValueError(f"{name} must be a non-empty list.")

        if not isinstance(field[0], list) or not field[0]:
            raise ValueError(f"{name} must contain non-empty grids.")

        if not isinstance(field[0][0], list) or not field[0][0]:
            raise ValueError(f"{name} must contain non-empty rows.")

        steps = len(field)
        ny = len(field[0])
        nx = len(field[0][0])

        validate_rectangular_field(field, [steps, ny, nx], name)
        return steps, ny, nx
