import json
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from django.test import SimpleTestCase

from port_twin_app.services.marine_ai.preparation_v2 import (
    _extract_windows,
    _load_scenario,
    _ScenarioData,
    prepare_marine_fno_v2_dataset,
)


def create_scenario(directory: Path, name: str, offset: float = 0.0):
    ny, nx = 3, 4
    times = np.arange(13, dtype=np.float64) * 3600.0
    mask = np.ones((ny, nx), dtype=bool)
    mask[0, 0] = False

    level = np.stack(
        [np.full((ny, nx), offset + index, dtype=np.float32) for index in range(13)]
    )
    current_u = np.stack(
        [
            np.full((ny, nx), offset + 0.1 * index, dtype=np.float32)
            for index in range(13)
        ]
    )
    current_v = np.stack(
        [
            np.full((ny, nx), offset + 0.2 * index, dtype=np.float32)
            for index in range(13)
        ]
    )

    dataset = xr.Dataset(
        {
            "water_level_m": (
                ("time_seconds", "northing", "easting"),
                level,
            ),
            "current_u_ms": (
                ("time_seconds", "northing", "easting"),
                current_u,
            ),
            "current_v_ms": (
                ("time_seconds", "northing", "easting"),
                current_v,
            ),
            "bathymetry_m": (
                ("northing", "easting"),
                np.full((ny, nx), 20.0, dtype=np.float32),
            ),
            "water_mask": (
                ("northing", "easting"),
                mask.astype(np.int8),
            ),
        },
        coords={
            "time_seconds": times,
            "easting": np.arange(nx, dtype=np.float64) * 250.0 + 500000.0,
            "northing": np.arange(ny, dtype=np.float64) * 250.0 + 9500000.0,
        },
        attrs={
            "dataset_version": "v2",
            "data_classification": "simulated",
            "is_observed": 0,
            "is_ground_truth": 0,
            "grid_crs": "EPSG:32737",
            "tidal_boundary": "east",
        },
    )

    dataset.to_netcdf(directory / f"{name}.nc", engine="h5netcdf")
    dataset.close()

    provenance = {
        "dataset_version": "v2",
        "data_classification": "simulated",
        "is_observed": False,
        "is_ground_truth": False,
        "configuration": {
            "wind_u_ms": 0.0,
            "wind_v_ms": 0.0,
            "tide_amplitude_m": 0.5,
            "tide_phase_radians": 0.0,
            "tide_period_seconds": 44712.0,
            "tidal_boundary": "east",
        },
    }
    (directory / f"{name}.provenance.json").write_text(
        json.dumps(provenance),
        encoding="utf-8",
    )


class MarineFNOV2PreparationTests(SimpleTestCase):
    def test_extracts_seven_six_hour_windows(self):
        mask = np.ones((2, 3), dtype=bool)
        scenario = _ScenarioData(
            name="v2_test",
            time_seconds=np.arange(13) * 3600.0,
            water_level=np.stack([np.full((2, 3), i) for i in range(13)]),
            current_u=np.stack([np.full((2, 3), i + 1) for i in range(13)]),
            current_v=np.stack([np.full((2, 3), i + 2) for i in range(13)]),
            bathymetry=np.full((2, 3), 20.0),
            water_mask=mask,
            easting=np.array([1.0, 2.0, 3.0]),
            northing=np.array([4.0, 5.0]),
            forcing={
                "wind_u_ms": 0.0,
                "wind_v_ms": 0.0,
                "tide_amplitude_m": 0.5,
                "tide_phase_radians": 0.0,
                "tide_period_seconds": 44712.0,
                "tidal_boundary": "east",
            },
        )

        samples = _extract_windows(scenario)

        self.assertEqual(len(samples), 7)
        self.assertEqual(samples[0].initial_state.shape, (3, 2, 3))
        self.assertEqual(samples[0].target.shape, (6, 3, 2, 3))
        self.assertEqual(samples[0].target[0, 0, 0, 0], 1.0)
        self.assertEqual(samples[-1].initial_state[0, 0, 0], 6.0)

    def test_masks_land_targets(self):
        mask = np.array([[False, True], [True, True]])
        scenario = _ScenarioData(
            name="v2_mask",
            time_seconds=np.arange(13) * 3600.0,
            water_level=np.ones((13, 2, 2)),
            current_u=np.ones((13, 2, 2)),
            current_v=np.ones((13, 2, 2)),
            bathymetry=np.full((2, 2), 20.0),
            water_mask=mask,
            easting=np.array([1.0, 2.0]),
            northing=np.array([3.0, 4.0]),
            forcing={},
        )

        sample = _extract_windows(scenario)[0]

        self.assertTrue(np.isnan(sample.target[:, :, 0, 0]).all())
        self.assertTrue(np.isfinite(sample.target[:, :, 1, 1]).all())

    def test_loads_v2_provenance_and_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            create_scenario(directory, "v2_test")

            scenario = _load_scenario(directory / "v2_test.nc")

            self.assertEqual(scenario.name, "v2_test")
            self.assertEqual(scenario.water_level.shape, (13, 3, 4))
            self.assertEqual(scenario.forcing["tidal_boundary"], "east")

    def test_rejects_wrong_dataset_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            create_scenario(directory, "v2_test")

            path = directory / "v2_test.nc"
            with xr.open_dataset(path) as source:
                dataset = source.load()

            dataset.attrs["dataset_version"] = "v1"
            dataset.to_netcdf(path, mode="w", engine="h5netcdf")
            dataset.close()

            with self.assertRaisesRegex(ValueError, "not a v2"):
                _load_scenario(path)

    def test_splits_all_scenarios_without_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)

            for index in range(18):
                create_scenario(directory, f"v2_scenario_{index:02d}", float(index))

            split = prepare_marine_fno_v2_dataset(directory)

            self.assertEqual(len(split.train), 84)
            self.assertEqual(len(split.validation), 21)
            self.assertEqual(len(split.test), 21)
            self.assertEqual(len(split.scenario_partitions), 18)

            self.assertEqual(
                set(split.scenario_partitions.values()),
                {"train", "validation", "test"},
            )

            for partition_name in ("train", "validation", "test"):
                samples = getattr(split, partition_name)
                scenario_ids = {sample.scenario_id for sample in samples}
                expected = {
                    name
                    for name, partition in split.scenario_partitions.items()
                    if partition == partition_name
                }
                self.assertEqual(scenario_ids, expected)
                self.assertEqual(len(scenario_ids), len(expected))
