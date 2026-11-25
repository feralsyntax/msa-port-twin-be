import math
from datetime import datetime, timedelta
from numbers import Real

from port_twin_app.models import MarinePrediction

FORECAST_STEPS = 6
FORECAST_STEP_SECONDS = 3600


def _validate_grid_frames(name, frames):
    """Validate a six-step grid forecast and return its dimensions."""

    if not isinstance(frames, list) or len(frames) != FORECAST_STEPS:
        raise ValueError(f"{name} must contain exactly six forecast frames.")

    dimensions = None

    for frame in frames:
        if not isinstance(frame, list) or not frame:
            raise ValueError(f"{name} contains an invalid forecast frame.")

        row_length = None

        for row in frame:
            if not isinstance(row, list) or not row:
                raise ValueError(f"{name} contains an invalid grid row.")

            if row_length is None:
                row_length = len(row)
            elif len(row) != row_length:
                raise ValueError(f"{name} contains inconsistent grid dimensions.")

            for value in row:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not math.isfinite(value)
                ):
                    raise ValueError(f"{name} contains a non-finite or invalid value.")

        frame_dimensions = (len(frame), row_length)

        if dimensions is None:
            dimensions = frame_dimensions
        elif frame_dimensions != dimensions:
            raise ValueError(f"{name} contains inconsistent forecast dimensions.")

    return dimensions


def _flatten(frame):
    return [value for row in frame for value in row]


def build_marine_condition_advisory(prediction):
    """
    Build hourly, grid-level marine condition summaries.

    These summaries describe the prediction grid as a whole.
    They do not represent route-level or navigational safety conditions.
    """

    if prediction.status != MarinePrediction.Status.COMPLETED:
        raise ValueError("Marine prediction must be completed.")

    if not isinstance(prediction.result_data, dict):
        raise ValueError("Marine prediction has no valid result data.")  # noqa: TRY004

    result = prediction.result_data

    water_levels = result.get("water_level_m")
    current_u = result.get("current_u_ms")
    current_v = result.get("current_v_ms")

    water_dimensions = _validate_grid_frames("water_level_m", water_levels)
    u_dimensions = _validate_grid_frames("current_u_ms", current_u)
    v_dimensions = _validate_grid_frames("current_v_ms", current_v)

    if water_dimensions != u_dimensions or water_dimensions != v_dimensions:
        raise ValueError("Marine prediction output grids have different dimensions.")

    grid = prediction.grid

    if grid.get("ny") != water_dimensions[0] or grid.get("nx") != water_dimensions[1]:
        raise ValueError("Marine prediction output dimensions do not match its grid.")

    forecast = prediction.forecast

    if (
        forecast.get("steps") != FORECAST_STEPS
        or forecast.get("step_seconds") != FORECAST_STEP_SECONDS
    ):
        raise ValueError("Marine prediction must contain six hourly forecast steps.")

    try:
        start_time = datetime.fromisoformat(
            forecast["start_time"].replace("Z", "+00:00")  # noqa: FURB162
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Marine prediction has an invalid forecast start time."
        ) from exc

    if start_time.tzinfo is None or start_time.utcoffset() is None:
        raise ValueError("Forecast start time must include a timezone.")

    hourly_conditions = []

    for index in range(FORECAST_STEPS):
        levels = _flatten(water_levels[index])
        u_values = _flatten(current_u[index])
        v_values = _flatten(current_v[index])

        current_speeds = [
            math.hypot(u, v) for u, v in zip(u_values, v_values, strict=True)
        ]

        interval_start = start_time + timedelta(seconds=index * FORECAST_STEP_SECONDS)
        interval_end = interval_start + timedelta(seconds=FORECAST_STEP_SECONDS)

        hourly_conditions.append(
            {
                "step": index + 1,
                "start_time": interval_start.isoformat(),
                "end_time": interval_end.isoformat(),
                "mean_water_level_m": math.fsum(levels) / len(levels),
                "minimum_water_level_m": min(levels),
                "maximum_water_level_m": max(levels),
                "mean_current_speed_ms": (
                    math.fsum(current_speeds) / len(current_speeds)
                ),
                "maximum_current_speed_ms": max(current_speeds),
            }
        )

    return {
        "prediction_id": str(prediction.id),
        "model_version": prediction.model_version,
        "contract_version": prediction.contract_version,
        "scope": "grid_level",
        "description": (
            "Hourly summaries across the prediction grid. "
            "These are not route-level conditions or navigational safety assessments."
        ),
        "hourly_conditions": hourly_conditions,
    }
