"""Preparation of normalized marine observations for FNO training."""

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import xarray as xr

from .contracts import (
    FORECAST_STEP_SECONDS,
    FORECAST_STEPS,
    MarineForecastConfig,
    MarineGrid,
    MarinePredictionRequest,
    MarinePredictionResult,
)
from .dataset import MarineDatasetSample, MarineDatasetSplit, split_dataset

MARINE_VARIABLES = (
    "sea_surface_height_m",
    "current_u_ms",
    "current_v_ms",
)

REQUIRED_BATHYMETRY_VARIABLE = "bathymetry_m"


@dataclass(frozen=True)
class PreparedMarineDataset:
    """Prepared samples, source metadata, and bathymetry quality summary."""

    split: MarineDatasetSplit
    metadata: dict
    tid_summary: dict[int, int]


def _require_variables(dataset, variables, name):
    missing = [variable for variable in variables if variable not in dataset]
    if missing:
        raise ValueError(f"{name} is missing variables: {missing}.")


def _validate_coordinates(dataset, name):
    for coordinate in ("latitude", "longitude"):
        if coordinate not in dataset.coords:
            raise ValueError(f"{name} is missing {coordinate} coordinates.")

        values = np.asarray(dataset[coordinate].values)

        if values.ndim != 1 or values.size < 2:
            raise ValueError(
                f"{name} {coordinate} must be a 1D grid with at least 2 points."
            )

        if not np.isfinite(values).all():
            raise ValueError(f"{name} {coordinate} contains non-finite values.")

        if not np.all(np.diff(values) > 0):
            raise ValueError(f"{name} {coordinate} must be strictly increasing.")


def _validate_grid_alignment(marine, bathymetry):
    for coordinate in ("latitude", "longitude"):
        marine_values = np.asarray(marine[coordinate].values)
        bathymetry_values = np.asarray(bathymetry[coordinate].values)

        if not np.array_equal(marine_values, bathymetry_values):
            raise ValueError(
                f"Marine and bathymetry {coordinate} coordinates must match exactly."
            )


def _validate_hourly_time(dataset):
    if "time" not in dataset.coords:
        raise ValueError("Marine dataset is missing time coordinates.")

    times = np.asarray(dataset["time"].values)

    if times.ndim != 1 or times.size < FORECAST_STEPS + 1:
        raise ValueError(
            f"At least {FORECAST_STEPS + 1} hourly timestamps are required."
        )

    if not np.issubdtype(times.dtype, np.datetime64):
        raise ValueError("Marine time coordinates must use datetime64 values.")

    if np.isnat(times).any():
        raise ValueError("Marine time coordinates contain invalid timestamps.")

    intervals = np.diff(times)

    if not np.all(intervals == np.timedelta64(FORECAST_STEP_SECONDS, "s")):
        raise ValueError("Marine observations must have continuous hourly intervals.")

    return times


def _validate_finite_values(dataset, variables, name):
    for variable in variables:
        values = np.asarray(dataset[variable].values)

        if not np.issubdtype(values.dtype, np.number):
            raise ValueError(f"{name} variable {variable} must be numeric.")

        if not np.isfinite(values).all():
            raise ValueError(
                f"{name} variable {variable} contains missing or non-finite values."
            )


def _validate_tid(bathymetry):
    if "tid" not in bathymetry:
        raise ValueError("GEBCO dataset must include the TID classification variable.")

    values = np.asarray(bathymetry["tid"].values)

    if values.shape != bathymetry["bathymetry_m"].shape:
        raise ValueError("TID and bathymetry must have matching dimensions.")

    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("TID classifications must be numeric.")

    if not np.isfinite(values).all():
        raise ValueError("TID classifications contain missing or non-finite values.")

    if not np.equal(values, np.floor(values)).all():
        raise ValueError("TID classifications must contain integer codes.")

    return values.astype(np.int64)


def _timestamp(value):
    """Convert a NumPy timestamp to a UTC-aware ISO 8601 string."""
    timestamp = np.datetime_as_string(value, unit="s")
    parsed = datetime.fromisoformat(timestamp)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)

    return parsed.isoformat()


def _source_metadata(marine, bathymetry):
    marine_attrs = marine.attrs
    bathymetry_attrs = bathymetry.attrs

    marine_version = str(
        marine_attrs.get("dataset_version")
        or marine_attrs.get("source_version")
        or "unspecified"
    )
    bathymetry_version = str(
        bathymetry_attrs.get("dataset_version")
        or bathymetry_attrs.get("source_version")
        or "unspecified"
    )

    marine_vertical_reference = marine_attrs.get("vertical_reference")
    bathymetry_vertical_reference = bathymetry_attrs.get("vertical_reference")

    return {
        "marine_source": marine_attrs.get("provider", "Copernicus Marine"),
        "marine_dataset_id": marine_attrs.get("dataset_id"),
        "marine_dataset_version": marine_version,
        "marine_license": marine_attrs.get(
            "license_name", "Copernicus Marine Service Licence"
        ),
        "bathymetry_source": bathymetry_attrs.get("provider", "GEBCO"),
        "bathymetry_dataset_id": bathymetry_attrs.get("source_dataset_id", "GEBCO"),
        "bathymetry_dataset_version": bathymetry_version,
        "bathymetry_license": bathymetry_attrs.get(
            "license", "GEBCO public domain; subject to its terms and disclaimer"
        ),
        "bathymetry_tid_source_file": bathymetry_attrs.get("tid_source_file_path"),
        "marine_vertical_reference": marine_vertical_reference,
        "bathymetry_vertical_reference": bathymetry_vertical_reference,
        "bathymetry_vertical_crs": bathymetry_attrs.get("vertical_crs"),
        "vertical_reference_warning": (
            "Marine water level and GEBCO elevation use separately documented "
            "vertical references. No datum transformation has been applied."
            if marine_vertical_reference != bathymetry_vertical_reference
            else None
        ),
    }


def prepare_marine_dataset(
    marine: xr.Dataset,
    bathymetry: xr.Dataset,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> PreparedMarineDataset:
    """Build six-hour supervised samples and split them chronologically."""

    if not isinstance(marine, xr.Dataset):
        raise ValueError("marine must be an xarray Dataset.")

    if not isinstance(bathymetry, xr.Dataset):
        raise ValueError("bathymetry must be an xarray Dataset.")

    _require_variables(marine, MARINE_VARIABLES, "Marine dataset")
    _require_variables(
        bathymetry, (REQUIRED_BATHYMETRY_VARIABLE,), "Bathymetry dataset"
    )

    _validate_coordinates(marine, "Marine dataset")
    _validate_coordinates(bathymetry, "Bathymetry dataset")
    _validate_grid_alignment(marine, bathymetry)

    times = _validate_hourly_time(marine)

    expected_dimensions = ("time", "latitude", "longitude")

    for variable in MARINE_VARIABLES:
        if marine[variable].dims != expected_dimensions:
            raise ValueError(f"{variable} must use dimensions {expected_dimensions}.")

    expected_bathymetry_dimensions = ("latitude", "longitude")

    if bathymetry[REQUIRED_BATHYMETRY_VARIABLE].dims != expected_bathymetry_dimensions:
        raise ValueError(
            f"bathymetry_m must use dimensions {expected_bathymetry_dimensions}."
        )

    _validate_finite_values(marine, MARINE_VARIABLES, "Marine dataset")
    _validate_finite_values(
        bathymetry, (REQUIRED_BATHYMETRY_VARIABLE,), "Bathymetry dataset"
    )

    bathymetry_values = np.asarray(bathymetry["bathymetry_m"].values)

    if np.any(bathymetry_values <= 0):
        raise ValueError("Bathymetry must contain positive underwater depths.")

    tid_values = _validate_tid(bathymetry)

    latitude = np.asarray(marine["latitude"].values, dtype=float)
    longitude = np.asarray(marine["longitude"].values, dtype=float)

    grid = MarineGrid(
        west=float(longitude[0]),
        south=float(latitude[0]),
        east=float(longitude[-1]),
        north=float(latitude[-1]),
        nx=len(longitude),
        ny=len(latitude),
    )

    marine_values = {
        variable: np.asarray(marine[variable].values, dtype=float)
        for variable in MARINE_VARIABLES
    }

    samples = []

    for index in range(len(times) - FORECAST_STEPS):
        start_time = _timestamp(times[index])

        request = MarinePredictionRequest(
            grid=grid,
            forecast=MarineForecastConfig(start_time=start_time),
            bathymetry_m=bathymetry_values.tolist(),
            initial_water_level_m=marine_values["sea_surface_height_m"][index].tolist(),
            initial_current_u_ms=marine_values["current_u_ms"][index].tolist(),
            initial_current_v_ms=marine_values["current_v_ms"][index].tolist(),
        )

        target_slice = slice(index + 1, index + FORECAST_STEPS + 1)

        target = MarinePredictionResult(
            water_level_m=marine_values["sea_surface_height_m"][target_slice].tolist(),
            current_u_ms=marine_values["current_u_ms"][target_slice].tolist(),
            current_v_ms=marine_values["current_v_ms"][target_slice].tolist(),
            model_version="copernicus-observation",
        )

        samples.append(
            MarineDatasetSample(
                sample_id=f"copernicus-{index:06d}",
                request=request,
                target=target,
                source="Copernicus Marine + GEBCO",
                source_version=(
                    f"Copernicus:{marine.attrs.get('dataset_version', 'unspecified')};"
                    f"GEBCO:{bathymetry.attrs.get('source_version', 'unspecified')}"
                ),
                license_name=(
                    "Copernicus Marine Service Licence; "
                    "GEBCO public domain subject to its terms and disclaimer"
                ),
            )
        )

    split = split_dataset(
        samples,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )

    tid_codes, tid_counts = np.unique(tid_values, return_counts=True)

    metadata = {
        **_source_metadata(marine, bathymetry),
        "grid": {
            "crs": "EPSG:4326",
            "nx": grid.nx,
            "ny": grid.ny,
            "west": grid.west,
            "east": grid.east,
            "south": grid.south,
            "north": grid.north,
        },
        "time_start": _timestamp(times[0]),
        "time_end": _timestamp(times[-1]),
        "time_steps": len(times),
        "forecast_steps": FORECAST_STEPS,
        "forecast_step_seconds": FORECAST_STEP_SECONDS,
        "generated_samples": len(samples),
        "train_samples": len(split.train),
        "validation_samples": len(split.validation),
        "test_samples": len(split.test),
        "purged_samples": len(samples) - split.total_samples,
        "marine_data_note": (
            "Copernicus Marine analysis and forecast fields are gridded model "
            "products, not independent in-situ observations."
        ),
        "sampling_note": (
            "Sliding forecast windows overlap and are temporally correlated."
        ),
    }

    return PreparedMarineDataset(
        split=split,
        metadata=metadata,
        tid_summary={
            int(code): int(count)
            for code, count in zip(tid_codes, tid_counts, strict=True)
        },
    )
