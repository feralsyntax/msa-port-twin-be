"""Prepare daily Copernicus reanalysis windows for the real-data marine FNO."""

import csv
from calendar import monthrange
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

FORECAST_STEPS = 6
START_YEAR = 2020
END_YEAR = 2025

TRAIN_YEARS = (2020, 2021, 2022, 2023)
VALIDATION_YEARS = (2024,)
TEST_YEARS = (2025,)

EXPECTED_DEPTH_M = 0.49402499198913574
DEPTH_TOLERANCE_M = 1e-4

VARIABLES = ("zos", "uo", "vo")

INPUT_CHANNEL_ORDER = (
    "initial_zos",
    "initial_uo",
    "initial_vo",
    "ocean_mask",
    "normalized_longitude",
    "normalized_latitude",
)

OUTPUT_CHANNEL_ORDER = ("zos", "uo", "vo")


@dataclass(frozen=True)
class MarineFNORealSample:
    """One initial ocean state and its following six daily targets."""

    sample_id: str
    start_time: np.datetime64
    inputs: np.ndarray
    target: np.ndarray


@dataclass(frozen=True)
class MarineFNORealSplit:
    """Chronologically separated samples for real-data FNO training."""

    train: tuple[MarineFNORealSample, ...]
    validation: tuple[MarineFNORealSample, ...]
    test: tuple[MarineFNORealSample, ...]
    longitude: np.ndarray
    latitude: np.ndarray
    water_mask: np.ndarray

    @property
    def total_samples(self) -> int:
        return len(self.train) + len(self.validation) + len(self.test)


def _expected_months() -> tuple[str, ...]:
    return tuple(
        f"{year}-{month:02d}"
        for year in range(START_YEAR, END_YEAR + 1)
        for month in range(1, 13)
    )


def _read_manifest(
    manifest_path: Path,
    project_root: Path,
) -> tuple[tuple[str, Path], ...]:
    """Validate the monthly manifest and locate its NetCDF files."""

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Copernicus manifest not found: {manifest_path}")

    with manifest_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"month", "start", "end", "time_steps", "source_file", "status"}

        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Manifest must contain columns: {sorted(required)}")

        records = list(reader)

    expected_months = _expected_months()
    actual_months = tuple(record["month"] for record in records)

    if actual_months != expected_months:
        raise ValueError(
            "Manifest must cover every month from 2020-01 to 2025-12 in order."
        )

    result = []

    for record in records:
        month = record["month"]
        year, month_number = map(int, month.split("-"))
        expected_start = f"{month}-01T00:00:00"

        if month_number == 12:
            expected_end = f"{year + 1}-01-01T00:00:00"
        else:
            expected_end = f"{year}-{month_number + 1:02d}-01T00:00:00"

        expected_steps = monthrange(year, month_number)[1]

        if record["status"] != "success":
            raise ValueError(f"Manifest month {month} is not marked successful.")

        if record["start"] != expected_start or record["end"] != expected_end:
            raise ValueError(f"Manifest month {month} has unexpected date boundaries.")

        try:
            time_steps = int(record["time_steps"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Manifest month {month} has invalid time_steps.") from exc

        if time_steps != expected_steps:
            raise ValueError(
                f"Manifest month {month} should contain {expected_steps} daily records."
            )

        source_path = Path(record["source_file"])
        if not source_path.is_absolute():
            source_path = project_root / source_path

        if not source_path.is_file():
            raise FileNotFoundError(
                f"NetCDF file for {month} does not exist: {source_path}"
            )

        result.append((month, source_path))

    return tuple(result)


def _load_month(
    path: Path,
    expected_month: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate one monthly Copernicus NetCDF file."""

    with xr.open_dataset(path) as source:
        dataset = source.load()

    try:
        missing_variables = set(VARIABLES).difference(dataset.data_vars)
        if missing_variables:
            raise ValueError(
                f"{path.name} is missing variables: {sorted(missing_variables)}"
            )

        required_coordinates = {"time", "latitude", "longitude", "depth"}
        missing_coordinates = required_coordinates.difference(dataset.coords)

        if missing_coordinates:
            raise ValueError(
                f"{path.name} is missing coordinates: {sorted(missing_coordinates)}"
            )

        if dataset.sizes.get("depth") != 1:
            raise ValueError(
                f"{path.name} must contain exactly one surface depth level."
            )

        depth = float(dataset["depth"].values[0])
        if not np.isclose(depth, EXPECTED_DEPTH_M, atol=DEPTH_TOLERANCE_M):
            raise ValueError(f"{path.name} has unexpected depth: {depth:.6f} m.")

        times = dataset["time"].values.astype("datetime64[ns]")
        longitude = np.asarray(dataset["longitude"].values, dtype=np.float64)
        latitude = np.asarray(dataset["latitude"].values, dtype=np.float64)

        if times.ndim != 1 or times.size == 0:
            raise ValueError(f"{path.name} has an invalid time coordinate.")

        if longitude.ndim != 1 or latitude.ndim != 1:
            raise ValueError(f"{path.name} must have one-dimensional coordinates.")

        if (
            not np.isfinite(longitude).all()
            or not np.isfinite(latitude).all()
            or np.any(np.diff(longitude) <= 0)
            or np.any(np.diff(latitude) <= 0)
        ):
            raise ValueError(f"{path.name} has invalid spatial coordinates.")

        expected_days = monthrange(*map(int, expected_month.split("-")))[1]
        expected_start = np.datetime64(f"{expected_month}-01", "D")
        expected_times = expected_start + np.arange(expected_days).astype(
            "timedelta64[D]"
        )

        if len(times) != expected_days or not np.array_equal(
            times.astype("datetime64[D]"), expected_times
        ):
            raise ValueError(
                f"{path.name} does not contain every day in {expected_month}."
            )

        expected_spatial_dims = ("latitude", "longitude")

        for variable in VARIABLES:
            dimensions = dataset[variable].dims

            if variable == "zos" and "depth" not in dimensions:
                expected_dimensions = ("time", *expected_spatial_dims)
                values = dataset[variable].transpose(*expected_dimensions).values
            else:
                expected_dimensions = ("time", "depth", *expected_spatial_dims)

                if dimensions != expected_dimensions:
                    raise ValueError(
                        f"{variable} in {path.name} must use dimensions "
                        f"{expected_dimensions}."
                    )

                values = (
                    dataset[variable]
                    .isel(depth=0)
                    .transpose("time", "latitude", "longitude")
                    .values
                )

            if values.shape != (len(times), len(latitude), len(longitude)):
                raise ValueError(f"{variable} in {path.name} has an invalid shape.")

            if variable == "zos":
                zos = values
            elif variable == "uo":
                uo = values
            else:
                vo = values

        fields = np.stack((zos, uo, vo), axis=1).astype(np.float32)

        return times, longitude, latitude, fields

    finally:
        dataset.close()


def _load_reanalysis(
    manifest_path: Path,
    project_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load all months and verify continuous dates and a stable ocean mask."""

    monthly_files = _read_manifest(manifest_path, project_root)

    time_parts = []
    field_parts = []
    reference_longitude = None
    reference_latitude = None

    for month, path in monthly_files:
        times, longitude, latitude, fields = _load_month(path, month)

        if reference_longitude is None:
            reference_longitude = longitude
            reference_latitude = latitude
        elif not (
            np.array_equal(longitude, reference_longitude)
            and np.array_equal(latitude, reference_latitude)
        ):
            raise ValueError(f"Spatial coordinates differ in {path.name}.")

        time_parts.append(times)
        field_parts.append(fields)

    times = np.concatenate(time_parts)
    fields = np.concatenate(field_parts, axis=0)

    expected_times = np.arange(
        np.datetime64("2020-01-01"),
        np.datetime64("2026-01-01"),
        np.timedelta64(1, "D"),
    ).astype("datetime64[ns]")

    if not np.array_equal(times, expected_times):
        raise ValueError(
            "Copernicus records must cover 2020-01-01 through 2025-12-31 "
            "with no missing or duplicate days."
        )

    valid_values = np.isfinite(fields)

    if not (
        np.array_equal(valid_values[:, 0], valid_values[:, 1])
        and np.array_equal(valid_values[:, 0], valid_values[:, 2])
    ):
        raise ValueError("zos, uo and vo must share the same ocean/land mask.")

    mask_by_day = valid_values[:, 0]
    water_mask = mask_by_day[0]

    if not water_mask.any():
        raise ValueError("Copernicus grid contains no valid ocean cells.")

    if not np.all(mask_by_day == water_mask[None, :, :]):
        raise ValueError(
            "Ocean mask changes over time; a stable land mask was expected."
        )

    if not np.isfinite(fields[:, :, water_mask]).all():
        raise ValueError("Ocean cells contain non-finite Copernicus values.")

    return times, reference_longitude, reference_latitude, fields, water_mask


def _coordinate_channels(
    longitude: np.ndarray,
    latitude: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Create longitude and latitude grids normalized to [-1, 1]."""

    normalized_longitude = (
        2.0 * (longitude - longitude.min()) / (longitude.max() - longitude.min()) - 1.0
    )
    normalized_latitude = (
        2.0 * (latitude - latitude.min()) / (latitude.max() - latitude.min()) - 1.0
    )

    longitude_grid, latitude_grid = np.meshgrid(
        normalized_longitude,
        normalized_latitude,
    )

    return longitude_grid.astype(np.float32), latitude_grid.astype(np.float32)


def _extract_partition_samples(
    times: np.ndarray,
    fields: np.ndarray,
    water_mask: np.ndarray,
    longitude_grid: np.ndarray,
    latitude_grid: np.ndarray,
) -> tuple[MarineFNORealSample, ...]:
    """Create six-day forecast samples without crossing year boundaries."""

    if len(times) <= FORECAST_STEPS:
        raise ValueError(
            "A partition must contain more days than the forecast horizon."
        )

    mask_channel = water_mask.astype(np.float32)
    samples = []

    for start in range(len(times) - FORECAST_STEPS):
        initial_state = fields[start]

        # The model must not receive NaN values for land cells.
        initial_inputs = np.where(water_mask[None, :, :], initial_state, 0.0)

        inputs = np.concatenate(
            (
                initial_inputs,
                mask_channel[None, :, :],
                longitude_grid[None, :, :],
                latitude_grid[None, :, :],
            ),
            axis=0,
        ).astype(np.float32, copy=False)

        # Keep land targets as NaN so training can exclude them from the loss.
        target = fields[start + 1 : start + 1 + FORECAST_STEPS].copy()

        start_time = times[start]
        date_label = np.datetime_as_string(start_time, unit="D")

        samples.append(
            MarineFNORealSample(
                sample_id=f"real_{date_label}",
                start_time=start_time,
                inputs=inputs,
                target=target,
            )
        )

    return tuple(samples)


def prepare_marine_fno_real_dataset(
    manifest_path: str | Path = (
        "data/raw/copernicus/mombasa_reanalysis_2020_2025_manifest.csv"
    ),
    project_root: str | Path | None = None,
) -> MarineFNORealSplit:
    """Prepare daily Copernicus windows with fixed chronological year splits."""

    root = Path(project_root) if project_root is not None else Path.cwd()
    manifest = Path(manifest_path)

    if not manifest.is_absolute():
        manifest = root / manifest

    times, longitude, latitude, fields, water_mask = _load_reanalysis(
        manifest,
        root,
    )

    longitude_grid, latitude_grid = _coordinate_channels(longitude, latitude)
    years = times.astype("datetime64[Y]").astype(int) + 1970

    partition_years = {
        "train": TRAIN_YEARS,
        "validation": VALIDATION_YEARS,
        "test": TEST_YEARS,
    }

    partitions = {}

    for name, selected_years in partition_years.items():
        selected = np.isin(years, selected_years)

        partitions[name] = _extract_partition_samples(
            times[selected],
            fields[selected],
            water_mask,
            longitude_grid,
            latitude_grid,
        )

    result = MarineFNORealSplit(
        train=partitions["train"],
        validation=partitions["validation"],
        test=partitions["test"],
        longitude=longitude.copy(),
        latitude=latitude.copy(),
        water_mask=water_mask.copy(),
    )

    expected_counts = {
        "train": 1455,
        "validation": 360,
        "test": 359,
    }

    actual_counts = {
        "train": len(result.train),
        "validation": len(result.validation),
        "test": len(result.test),
    }

    if actual_counts != expected_counts:
        raise ValueError(
            f"Unexpected forecast-window counts: {actual_counts}; "
            f"expected {expected_counts}."
        )

    return result
