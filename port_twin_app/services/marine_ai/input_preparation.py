"""Prepare validated, dataset-backed inputs for marine FNO inference."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from django.conf import settings
from django.utils import timezone

from port_twin_app.services.marine_ai.contracts import (
    MarineForecastConfig,
    MarineGrid,
    MarinePredictionRequest,
)
from port_twin_app.services.marine_data.normalization import (
    normalize_copernicus_dataset,
    normalize_gebco_bathymetry,
)
from port_twin_app.services.marine_data.validation import (
    validate_bathymetry_dataset,
    validate_marine_dataset,
)

DOMAIN = "mombasa_offshore"

GRID_BOUNDS = {
    "west": 40.16667175292969,
    "south": -4.916664123535156,
    "east": 40.91667175292969,
    "north": -3.0833282470703125,
}

GRID_NX = 10
GRID_NY = 23


class MarineInputUnavailable(Exception):
    """Raised when configured marine inputs cannot support a prediction."""


@dataclass(frozen=True)
class PreparedMarineInputs:
    prediction_request: MarinePredictionRequest
    grid_data: dict
    forecast_data: dict
    input_data: dict


def _configured_path(setting_name):
    value = getattr(settings, setting_name, None)

    if not value:
        raise MarineInputUnavailable(f"{setting_name} is not configured.")

    path = Path(value)

    if not path.is_file():
        raise MarineInputUnavailable(
            f"Configured marine dataset does not exist: {path}"
        )

    return path


def _read_provenance(dataset_path):
    provenance_path = dataset_path.with_suffix(".provenance.json")

    if not provenance_path.is_file():
        raise MarineInputUnavailable(
            f"Missing provenance sidecar for {dataset_path.name}."
        )

    try:
        with provenance_path.open(encoding="utf-8") as file:
            provenance = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise MarineInputUnavailable(
            f"Could not read provenance for {dataset_path.name}."
        ) from exc

    required_fields = (
        "provider",
        "dataset_id",
        "dataset_version",
        "retrieved_at",
    )

    if any(not provenance.get(field) for field in required_fields):
        raise MarineInputUnavailable(f"Incomplete provenance for {dataset_path.name}.")

    return provenance


def _parse_datetime(value, description):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))  # noqa: FURB162
    except (AttributeError, TypeError, ValueError) as exc:
        raise MarineInputUnavailable(f"Invalid {description} timestamp.") from exc

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise MarineInputUnavailable(
            f"{description} timestamp must include a timezone."
        )

    return parsed.astimezone(UTC)


def _check_freshness(provenance, dataset_name):
    retrieved_at = _parse_datetime(
        provenance["retrieved_at"],
        f"{dataset_name} retrieval",
    )

    max_age_hours = getattr(
        settings,
        "MARINE_DATA_MAX_AGE_HOURS",
        3,
    )

    if max_age_hours <= 0:
        raise MarineInputUnavailable("MARINE_DATA_MAX_AGE_HOURS must be positive.")

    age = timezone.now() - retrieved_at

    if age > timedelta(hours=max_age_hours):
        raise MarineInputUnavailable(
            f"{dataset_name} dataset is stale. "
            f"It was retrieved at {retrieved_at.isoformat()}."
        )

    if age < timedelta(0):
        raise MarineInputUnavailable(
            f"{dataset_name} retrieval timestamp is in the future."
        )


def _check_spatial_coverage(dataset, dataset_name):
    latitude = dataset.latitude.values
    longitude = dataset.longitude.values

    if (
        GRID_BOUNDS["south"] < float(latitude.min())
        or GRID_BOUNDS["north"] > float(latitude.max())
        or GRID_BOUNDS["west"] < float(longitude.min())
        or GRID_BOUNDS["east"] > float(longitude.max())
    ):
        raise MarineInputUnavailable(
            f"{dataset_name} does not cover the configured offshore grid."
        )


def _grid_coordinates():
    longitude = np.linspace(
        GRID_BOUNDS["west"],
        GRID_BOUNDS["east"],
        GRID_NX,
    )
    latitude = np.linspace(
        GRID_BOUNDS["south"],
        GRID_BOUNDS["north"],
        GRID_NY,
    )

    return latitude, longitude


def _interpolate(dataset, latitude, longitude, variables):
    interpolated = dataset[list(variables)].interp(
        latitude=latitude,
        longitude=longitude,
        method="linear",
    )

    arrays = {}

    for variable in variables:
        values = np.asarray(interpolated[variable].values, dtype=np.float64)

        if not np.isfinite(values).all():
            raise MarineInputUnavailable(
                f"Interpolation produced missing or non-finite values for {variable}."
            )

        arrays[variable] = values

    return arrays


def prepare_marine_prediction_inputs(start_time, domain=DOMAIN):
    """
    Assemble the internal FNO request from configured Copernicus and GEBCO data.

    The requested timestamp must exist exactly in the Copernicus dataset.
    No temporal interpolation or fallback is performed.
    """
    if domain != DOMAIN:
        raise ValueError(f"Unsupported marine prediction domain: {domain}")

    requested_time = _parse_datetime(start_time, "Forecast start")

    copernicus_path = _configured_path("MARINE_COPERNICUS_DATASET_PATH")
    gebco_path = _configured_path("MARINE_GEBCO_DATASET_PATH")

    copernicus_provenance = _read_provenance(copernicus_path)
    gebco_provenance = _read_provenance(gebco_path)

    _check_freshness(copernicus_provenance, "Copernicus")

    try:
        with xr.open_dataset(copernicus_path) as raw_marine:
            marine = normalize_copernicus_dataset(raw_marine.load())

        with xr.open_dataset(gebco_path) as raw_bathymetry:
            raw_bathymetry = raw_bathymetry.load()

            gebco_renames = {}

            if "lat" in raw_bathymetry.dims or "lat" in raw_bathymetry.coords:
                gebco_renames["lat"] = "latitude"

            if "lon" in raw_bathymetry.dims or "lon" in raw_bathymetry.coords:
                gebco_renames["lon"] = "longitude"

            if "elevation" in raw_bathymetry.data_vars:
                gebco_renames["elevation"] = "elevation_m"

            if gebco_renames:
                raw_bathymetry = raw_bathymetry.rename(gebco_renames)

            bathymetry = normalize_gebco_bathymetry(raw_bathymetry)

        validate_marine_dataset(marine)
        validate_bathymetry_dataset(bathymetry)

    except MarineInputUnavailable:
        raise
    except Exception as exc:
        raise MarineInputUnavailable(
            "Could not load or validate configured marine datasets."
        ) from exc

    _check_spatial_coverage(marine, "Copernicus")
    _check_spatial_coverage(bathymetry, "GEBCO")

    # Dataset timestamps are stored as UTC-naive datetime64 values.
    requested_time_naive = requested_time.replace(tzinfo=None)
    requested_time64 = np.datetime64(requested_time_naive, "ns")

    available_times = marine.time.values

    if not np.any(available_times == requested_time64):
        raise MarineInputUnavailable(
            "No Copernicus observation exists at the requested "
            f"initialisation time: {requested_time.isoformat()}."
        )

    marine_at_start = marine.sel(time=requested_time64)

    latitude, longitude = _grid_coordinates()

    marine_arrays = _interpolate(
        marine_at_start,
        latitude,
        longitude,
        (
            "sea_surface_height_m",
            "current_u_ms",
            "current_v_ms",
        ),
    )

    bathymetry_arrays = _interpolate(
        bathymetry,
        latitude,
        longitude,
        ("bathymetry_m",),
    )

    bathymetry_values = bathymetry_arrays["bathymetry_m"]

    if np.any(bathymetry_values <= 0):
        raise MarineInputUnavailable(
            "The offshore grid contains invalid or non-positive bathymetry."
        )

    grid = MarineGrid(
        **GRID_BOUNDS,
        nx=GRID_NX,
        ny=GRID_NY,
    )

    forecast = MarineForecastConfig(start_time=requested_time.isoformat())

    prediction_request = MarinePredictionRequest(
        grid=grid,
        forecast=forecast,
        bathymetry_m=bathymetry_values.tolist(),
        initial_water_level_m=(marine_arrays["sea_surface_height_m"].tolist()),
        initial_current_u_ms=marine_arrays["current_u_ms"].tolist(),
        initial_current_v_ms=marine_arrays["current_v_ms"].tolist(),
    )

    grid_data = {
        **GRID_BOUNDS,
        "nx": GRID_NX,
        "ny": GRID_NY,
        "crs": "EPSG:4326",
        "domain": DOMAIN,
    }

    forecast_data = {
        "start_time": requested_time.isoformat(),
        "steps": forecast.steps,
        "step_seconds": forecast.step_seconds,
    }

    input_data = {
        "domain": DOMAIN,
        "grid": grid_data,
        "source_data": {
            "marine": {
                **copernicus_provenance,
                "selected_time": requested_time.isoformat(),
                "selected_variables": [
                    "zos",
                    "uo",
                    "vo",
                ],
            },
            "bathymetry": {
                **gebco_provenance,
                "selected_variable": "elevation",
            },
        },
    }

    return PreparedMarineInputs(
        prediction_request=prediction_request,
        grid_data=grid_data,
        forecast_data=forecast_data,
        input_data=input_data,
    )
