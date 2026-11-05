"""Serialization and loading of prepared marine AI datasets."""

import json
from pathlib import Path

import numpy as np

from .contracts import (
    CONTRACT_VERSION,
    MarineForecastConfig,
    MarineGrid,
    MarinePredictionRequest,
    MarinePredictionResult,
)
from .dataset import MarineDatasetSample, MarineDatasetSplit
from .preparation import PreparedMarineDataset

FORMAT_VERSION = 1
PARTITIONS = ("train", "validation", "test")


def save_prepared_dataset(
    prepared: PreparedMarineDataset,
    archive_path: str | Path,
) -> tuple[Path, Path]:
    """Save a prepared dataset as a compressed NumPy archive and JSON manifest."""

    if not isinstance(prepared, PreparedMarineDataset):
        raise TypeError("prepared must be a PreparedMarineDataset instance.")

    if not isinstance(prepared.split, MarineDatasetSplit):
        raise TypeError("prepared.split must be a MarineDatasetSplit instance.")

    archive_path = Path(archive_path)
    metadata_path = archive_path.with_suffix(".json")
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    partitions = {
        partition: getattr(prepared.split, partition) for partition in PARTITIONS
    }

    if any(not samples for samples in partitions.values()):
        raise ValueError("All dataset partitions must contain samples.")

    samples = tuple(
        sample for partition in PARTITIONS for sample in partitions[partition]
    )

    if any(not isinstance(sample, MarineDatasetSample) for sample in samples):
        raise ValueError("All samples must be MarineDatasetSample instances.")

    first = samples[0]
    grid = first.request.grid
    forecast = first.request.forecast

    bathymetry = np.asarray(first.request.bathymetry_m, dtype=np.float64)

    initial_states = []
    targets = []
    sample_records = []
    partition_labels = []

    for partition, partition_samples in partitions.items():
        for sample in partition_samples:
            request = sample.request

            if request.grid != grid:
                raise ValueError("All samples must use the same grid.")

            if request.forecast.steps != forecast.steps:
                raise ValueError("All samples must use the same forecast horizon.")

            if request.forecast.step_seconds != forecast.step_seconds:
                raise ValueError("All samples must use the same forecast interval.")

            if not np.array_equal(
                np.asarray(request.bathymetry_m, dtype=np.float64),
                bathymetry,
            ):
                raise ValueError("All samples must use identical bathymetry.")

            initial_states.append(
                np.stack(
                    (
                        request.initial_water_level_m,
                        request.initial_current_u_ms,
                        request.initial_current_v_ms,
                    )
                )
            )

            targets.append(
                np.stack(
                    (
                        sample.target.water_level_m,
                        sample.target.current_u_ms,
                        sample.target.current_v_ms,
                    ),
                    axis=1,
                )
            )

            sample_records.append(
                {
                    "sample_id": sample.sample_id,
                    "source": sample.source,
                    "source_version": sample.source_version,
                    "license_name": sample.license_name,
                    "target_model_version": sample.target.model_version,
                    "target_contract_version": sample.target.contract_version,
                }
            )

            partition_labels.append(partition)

    sample_ids = [sample.sample_id for sample in samples]

    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Sample IDs must be unique.")

    arrays = {
        "bathymetry_m": bathymetry,
        "initial_states": np.asarray(initial_states, dtype=np.float64),
        "targets": np.asarray(targets, dtype=np.float64),
        "sample_ids": np.asarray(sample_ids, dtype=np.str_),
        "start_times": np.asarray(
            [sample.request.forecast.start_time for sample in samples],
            dtype=np.str_,
        ),
        "partitions": np.asarray(partition_labels, dtype=np.str_),
    }

    for name, array in arrays.items():
        if name not in ("sample_ids", "start_times", "partitions"):  # noqa: SIM102
            if not np.isfinite(array).all():
                raise ValueError(f"Cannot save non-finite values in {name}.")

    manifest = {
        "format_version": FORMAT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "grid": {
            "west": grid.west,
            "south": grid.south,
            "east": grid.east,
            "north": grid.north,
            "nx": grid.nx,
            "ny": grid.ny,
            "crs": grid.crs,
        },
        "forecast": {
            "steps": forecast.steps,
            "step_seconds": forecast.step_seconds,
        },
        "metadata": prepared.metadata,
        "tid_summary": {
            str(code): count for code, count in prepared.tid_summary.items()
        },
        "sample_records": sample_records,
        "partition_counts": {
            partition: len(partitions[partition]) for partition in PARTITIONS
        },
    }

    # Write the archive through a file handle to avoid automatic suffix changes.
    temporary_archive = archive_path.with_name(f"{archive_path.name}.tmp")

    with temporary_archive.open("wb") as file:
        np.savez_compressed(file, **arrays)

    temporary_archive.replace(archive_path)

    temporary_metadata = metadata_path.with_name(f"{metadata_path.name}.tmp")

    temporary_metadata.write_text(
        json.dumps(manifest, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary_metadata.replace(metadata_path)

    return archive_path, metadata_path


def load_prepared_dataset(
    archive_path: str | Path,
) -> PreparedMarineDataset:
    """Load and reconstruct a previously saved prepared marine dataset."""

    archive_path = Path(archive_path)
    metadata_path = archive_path.with_suffix(".json")

    if not archive_path.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {archive_path}")

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")

    manifest = json.loads(metadata_path.read_text(encoding="utf-8"))

    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported dataset serialization format version.")

    if manifest.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("Dataset prediction contract version is incompatible.")

    with np.load(archive_path, allow_pickle=False) as archive:
        required_arrays = {
            "bathymetry_m",
            "initial_states",
            "targets",
            "sample_ids",
            "start_times",
            "partitions",
        }

        missing = required_arrays.difference(archive.files)

        if missing:
            raise ValueError(f"Dataset archive is missing arrays: {sorted(missing)}.")

        arrays = {name: archive[name] for name in required_arrays}

    bathymetry = arrays["bathymetry_m"]
    initial_states = arrays["initial_states"]
    targets = arrays["targets"]
    sample_ids = arrays["sample_ids"]
    start_times = arrays["start_times"]
    partitions = arrays["partitions"]

    if bathymetry.ndim != 2:
        raise ValueError("Saved bathymetry must have shape [ny, nx].")

    ny, nx = bathymetry.shape

    if initial_states.ndim != 4 or initial_states.shape[1:] != (3, ny, nx):
        raise ValueError("Saved initial states have an invalid shape.")

    sample_count = initial_states.shape[0]

    if targets.shape != (sample_count, 6, 3, ny, nx):
        raise ValueError("Saved targets have an invalid shape.")

    for name, values in (
        ("sample_ids", sample_ids),
        ("start_times", start_times),
        ("partitions", partitions),
    ):
        if values.shape != (sample_count,):
            raise ValueError(f"Saved {name} have an invalid shape.")

    for name, values in (
        ("bathymetry_m", bathymetry),
        ("initial_states", initial_states),
        ("targets", targets),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"Saved {name} contain non-finite values.")

    if len(set(sample_ids.tolist())) != sample_count:
        raise ValueError("Saved sample IDs are not unique.")

    if not set(partitions.tolist()).issubset(PARTITIONS):
        raise ValueError("Saved dataset contains an unknown partition.")

    grid_data = manifest["grid"]
    forecast_data = manifest["forecast"]

    grid = MarineGrid(**grid_data)
    forecast_steps = forecast_data["steps"]
    forecast_interval = forecast_data["step_seconds"]

    if (grid.ny, grid.nx) != (ny, nx):
        raise ValueError("Saved grid dimensions do not match the arrays.")

    if len(manifest["sample_records"]) != sample_count:
        raise ValueError("Saved sample metadata count does not match the arrays.")

    records = manifest["sample_records"]
    reconstructed = {partition: [] for partition in PARTITIONS}

    for index in range(sample_count):
        record = records[index]

        if record["sample_id"] != sample_ids[index]:
            raise ValueError("Sample IDs do not match their metadata records.")

        request = MarinePredictionRequest(
            grid=grid,
            forecast=MarineForecastConfig(
                start_time=str(start_times[index]),
                steps=forecast_steps,
                step_seconds=forecast_interval,
            ),
            bathymetry_m=bathymetry.tolist(),
            initial_water_level_m=initial_states[index, 0].tolist(),
            initial_current_u_ms=initial_states[index, 1].tolist(),
            initial_current_v_ms=initial_states[index, 2].tolist(),
        )

        target = MarinePredictionResult(
            water_level_m=targets[index, :, 0].tolist(),
            current_u_ms=targets[index, :, 1].tolist(),
            current_v_ms=targets[index, :, 2].tolist(),
            model_version=record["target_model_version"],
            contract_version=record["target_contract_version"],
        )

        sample = MarineDatasetSample(
            sample_id=record["sample_id"],
            request=request,
            target=target,
            source=record["source"],
            source_version=record["source_version"],
            license_name=record["license_name"],
        )

        reconstructed[str(partitions[index])].append(sample)

    split = MarineDatasetSplit(
        train=tuple(reconstructed["train"]),
        validation=tuple(reconstructed["validation"]),
        test=tuple(reconstructed["test"]),
    )

    for partition in PARTITIONS:
        expected = manifest["partition_counts"][partition]

        if len(reconstructed[partition]) != expected:
            raise ValueError(
                f"Saved {partition} partition count does not match its manifest."
            )

    if any(not getattr(split, partition) for partition in PARTITIONS):
        raise ValueError("Loaded dataset contains an empty partition.")

    return PreparedMarineDataset(
        split=split,
        metadata=manifest["metadata"],
        tid_summary={
            int(code): int(count) for code, count in manifest["tid_summary"].items()
        },
    )
