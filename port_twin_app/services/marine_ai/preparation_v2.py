"""Dataset preparation for the simulated, forcing-conditioned marine FNO v2."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

FORECAST_STEPS = 6
OUTPUT_INTERVAL_SECONDS = 3600.0
EXPECTED_SNAPSHOTS = 13
EXPECTED_DATASET_VERSION = "v2"


@dataclass(frozen=True)
class MarineFNOV2Sample:
    """One six-hour forecast window from a single simulation scenario."""

    sample_id: str
    scenario_id: str
    start_time_seconds: float
    initial_state: np.ndarray
    bathymetry_m: np.ndarray
    water_mask: np.ndarray
    easting_m: np.ndarray
    northing_m: np.ndarray
    target: np.ndarray
    forcing: dict


@dataclass(frozen=True)
class MarineFNOV2Split:
    """Scenario-isolated training, validation and test samples."""

    train: tuple[MarineFNOV2Sample, ...]
    validation: tuple[MarineFNOV2Sample, ...]
    test: tuple[MarineFNOV2Sample, ...]
    scenario_partitions: dict[str, str]

    @property
    def total_samples(self):
        return len(self.train) + len(self.validation) + len(self.test)


@dataclass(frozen=True)
class _ScenarioData:
    """Validated simulation arrays shared by its forecast windows."""

    name: str
    time_seconds: np.ndarray
    water_level: np.ndarray
    current_u: np.ndarray
    current_v: np.ndarray
    bathymetry: np.ndarray
    water_mask: np.ndarray
    easting: np.ndarray
    northing: np.ndarray
    forcing: dict


def _read_forcing(attrs: dict, provenance_path: Path) -> dict:
    """Read forcing values from the scenario provenance manifest."""

    if not provenance_path.is_file():
        raise ValueError(f"Missing scenario provenance: {provenance_path}")

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    if provenance.get("dataset_version") != EXPECTED_DATASET_VERSION:
        raise ValueError("Provenance does not identify dataset version v2.")

    if provenance.get("data_classification") != "simulated":
        raise ValueError("Scenario provenance must classify data as simulated.")

    if provenance.get("is_observed") is not False:
        raise ValueError("Scenario provenance must declare is_observed=false.")

    if provenance.get("is_ground_truth") is not False:
        raise ValueError("Scenario provenance must declare is_ground_truth=false.")

    config = provenance.get("configuration")
    if not isinstance(config, dict):
        raise ValueError("Scenario provenance is missing its configuration.")  # noqa: TRY004

    required = (
        "wind_u_ms",
        "wind_v_ms",
        "tide_amplitude_m",
        "tide_phase_radians",
        "tide_period_seconds",
        "tidal_boundary",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Scenario forcing configuration missing: {missing}")

    forcing = {key: config[key] for key in required}

    for key in required[:-1]:
        value = forcing[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Invalid numeric forcing value: {key}")  # noqa: TRY004
        if not np.isfinite(value):
            raise ValueError(f"Non-finite forcing value: {key}")

    if forcing["tide_period_seconds"] <= 0:
        raise ValueError("Tide period must be positive.")

    if forcing["tidal_boundary"] != "east":
        raise ValueError("FNO v2 expects tidal forcing at the east boundary.")

    if attrs.get("tidal_boundary") != forcing["tidal_boundary"]:
        raise ValueError("NetCDF and provenance tidal boundaries do not match.")

    return forcing


def _load_scenario(path: Path) -> _ScenarioData:
    """Load and validate one complete v2 simulation file."""

    if not path.is_file() or path.suffix != ".nc":
        raise ValueError(f"Expected an existing NetCDF scenario file: {path}")

    provenance_path = path.with_name(f"{path.stem}.provenance.json")

    with xr.open_dataset(path) as source:
        dataset = source.load()

    try:
        attrs = dataset.attrs

        if attrs.get("dataset_version") != EXPECTED_DATASET_VERSION:
            raise ValueError(f"{path.name} is not a v2 dataset.")

        if attrs.get("data_classification") != "simulated":
            raise ValueError("Scenario must be classified as simulated.")

        if attrs.get("is_observed") not in (False, 0):
            raise ValueError("Scenario must declare is_observed=false.")

        if attrs.get("is_ground_truth") not in (False, 0):
            raise ValueError("Scenario must declare is_ground_truth=false.")

        expected_variables = {
            "water_level_m",
            "current_u_ms",
            "current_v_ms",
            "bathymetry_m",
            "water_mask",
        }
        missing = expected_variables - set(dataset.data_vars)
        if missing:
            raise ValueError(f"Scenario is missing variables: {sorted(missing)}")

        expected_dims = ("time_seconds", "northing", "easting")
        for name in ("water_level_m", "current_u_ms", "current_v_ms"):
            if dataset[name].dims != expected_dims:
                raise ValueError(f"{name} has unexpected dimensions.")

        spatial_dims = ("northing", "easting")
        for name in ("bathymetry_m", "water_mask"):
            if dataset[name].dims != spatial_dims:
                raise ValueError(f"{name} has unexpected dimensions.")

        times = dataset.time_seconds.values.astype(np.float64)
        if times.shape != (EXPECTED_SNAPSHOTS,):
            raise ValueError("Expected exactly 13 hourly simulation snapshots.")

        if not np.isfinite(times).all() or not np.allclose(
            np.diff(times),
            OUTPUT_INTERVAL_SECONDS,
        ):
            raise ValueError("Simulation snapshots must be hourly and finite.")

        if not np.isclose(times[0], 0.0):
            raise ValueError("Simulation must begin at t=0.")

        if not np.isclose(times[-1], 12 * OUTPUT_INTERVAL_SECONDS):
            raise ValueError("Simulation must end at 12 hours.")

        easting = dataset.easting.values.astype(np.float64)
        northing = dataset.northing.values.astype(np.float64)

        if (
            easting.ndim != 1
            or northing.ndim != 1
            or len(easting) < 2
            or len(northing) < 2
            or not np.isfinite(easting).all()
            or not np.isfinite(northing).all()
            or np.any(np.diff(easting) <= 0)
            or np.any(np.diff(northing) <= 0)
        ):
            raise ValueError("Invalid projected grid coordinates.")

        if attrs.get("grid_crs") != "EPSG:32737":
            raise ValueError("Expected projected grid CRS EPSG:32737.")

        bathymetry = dataset.bathymetry_m.values.astype(np.float32)
        water_mask = dataset.water_mask.values.astype(bool)

        expected_shape = (len(northing), len(easting))
        if bathymetry.shape != expected_shape or water_mask.shape != expected_shape:
            raise ValueError("Bathymetry or water mask does not match grid shape.")

        if not water_mask.any():
            raise ValueError("Scenario contains no valid water cells.")

        if not np.isfinite(bathymetry[water_mask]).all():
            raise ValueError("Bathymetry contains invalid values in water cells.")

        if np.any(bathymetry[water_mask] <= 0):
            raise ValueError("Water-cell bathymetry must be positive downward.")

        fields = []
        for name in ("water_level_m", "current_u_ms", "current_v_ms"):
            values = dataset[name].values.astype(np.float32)

            if values.shape != (
                EXPECTED_SNAPSHOTS,
                *expected_shape,
            ):
                raise ValueError(f"{name} has an unexpected shape.")

            if not np.isfinite(values[:, water_mask]).all():
                raise ValueError(f"{name} contains non-finite water-cell values.")

            fields.append(values)

        forcing = _read_forcing(attrs, provenance_path)

        return _ScenarioData(
            name=path.stem,
            time_seconds=times,
            water_level=fields[0],
            current_u=fields[1],
            current_v=fields[2],
            bathymetry=bathymetry,
            water_mask=water_mask,
            easting=easting,
            northing=northing,
            forcing=forcing,
        )
    finally:
        dataset.close()


def _extract_windows(scenario: _ScenarioData) -> tuple[MarineFNOV2Sample, ...]:
    """Extract seven overlapping six-hour windows from a 12-hour run."""

    samples = []

    for start in range(EXPECTED_SNAPSHOTS - FORECAST_STEPS):
        start_time_seconds = float(scenario.time_seconds[start])

        initial_state = np.stack(
            (
                scenario.water_level[start],
                scenario.current_u[start],
                scenario.current_v[start],
            ),
            axis=0,
        )

        target = np.stack(
            (
                scenario.water_level[start + 1 : start + 1 + FORECAST_STEPS],
                scenario.current_u[start + 1 : start + 1 + FORECAST_STEPS],
                scenario.current_v[start + 1 : start + 1 + FORECAST_STEPS],
            ),
            axis=1,
        ).astype(np.float32, copy=False)

        # Land and disconnected cells are excluded from supervised targets.
        target = target.copy()
        target[:, :, ~scenario.water_mask] = np.nan

        # Advance the tidal phase to the beginning of this forecast window.
        # This follows the solver convention:
        # tide(t) = amplitude * sin(2*pi*t/period + initial_phase).
        forcing = dict(scenario.forcing)
        forcing["tide_phase_radians"] = (
            forcing["tide_phase_radians"]
            + 2.0 * np.pi * start_time_seconds / forcing["tide_period_seconds"]
        ) % (2.0 * np.pi)

        samples.append(
            MarineFNOV2Sample(
                sample_id=f"{scenario.name}_t{start:02d}",
                scenario_id=scenario.name,
                start_time_seconds=start_time_seconds,
                initial_state=initial_state,
                bathymetry_m=scenario.bathymetry.copy(),
                water_mask=scenario.water_mask.copy(),
                easting_m=scenario.easting.copy(),
                northing_m=scenario.northing.copy(),
                target=target,
                forcing=forcing,
            )
        )

    return tuple(samples)


def prepare_marine_fno_v2_dataset(
    dataset_directory: str | Path,
    seed: int = 42,
) -> MarineFNOV2Split:
    """Load all v2 runs, then split complete scenarios 12/3/3."""

    directory = Path(dataset_directory)
    paths = sorted(directory.glob("v2_*.nc"))

    if len(paths) != 18:
        raise ValueError(f"Expected 18 v2 scenario files, found {len(paths)}.")

    scenarios = tuple(_load_scenario(path) for path in paths)

    reference = scenarios[0]
    for scenario in scenarios[1:]:
        if not np.array_equal(scenario.easting, reference.easting):
            raise ValueError("Scenario easting coordinates do not match.")
        if not np.array_equal(scenario.northing, reference.northing):
            raise ValueError("Scenario northing coordinates do not match.")
        if not np.array_equal(
            scenario.bathymetry, reference.bathymetry, equal_nan=True
        ):
            raise ValueError("Scenario bathymetry does not match.")
        if not np.array_equal(scenario.water_mask, reference.water_mask):
            raise ValueError("Scenario water masks do not match.")

    names = np.asarray([scenario.name for scenario in scenarios])
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(names)

    train_names = set(shuffled[:12])
    validation_names = set(shuffled[12:15])

    partitions = {}
    samples_by_partition = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for scenario in scenarios:
        if scenario.name in train_names:
            partition = "train"
        elif scenario.name in validation_names:
            partition = "validation"
        else:
            partition = "test"

        partitions[scenario.name] = partition
        samples_by_partition[partition].extend(_extract_windows(scenario))

    return MarineFNOV2Split(
        train=tuple(samples_by_partition["train"]),
        validation=tuple(samples_by_partition["validation"]),
        test=tuple(samples_by_partition["test"]),
        scenario_partitions=partitions,
    )
