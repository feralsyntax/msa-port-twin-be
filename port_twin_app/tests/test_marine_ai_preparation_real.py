import csv
import tempfile
from calendar import monthrange
from pathlib import Path

import numpy as np
import xarray as xr
from django.test import SimpleTestCase

from port_twin_app.services.marine_ai.preparation_real import (
    FORECAST_STEPS,
    _coordinate_channels,
    _extract_partition_samples,
    _load_month,
    _read_manifest,
)


class RealMarinePreparationTests(SimpleTestCase):
    def test_coordinate_channels_are_normalized_to_minus_one_and_one(self):
        longitude = np.array([40.0, 40.5, 41.0])
        latitude = np.array([-5.0, -4.5])

        longitude_grid, latitude_grid = _coordinate_channels(
            longitude,
            latitude,
        )

        self.assertEqual(longitude_grid.shape, (2, 3))
        self.assertEqual(latitude_grid.shape, (2, 3))

        np.testing.assert_allclose(
            longitude_grid[0],
            [-1.0, 0.0, 1.0],
        )
        np.testing.assert_allclose(
            latitude_grid[:, 0],
            [-1.0, 1.0],
        )

    def test_extracts_six_day_forecast_windows(self):
        times = np.datetime64("2020-01-01") + np.arange(10) * np.timedelta64(1, "D")

        fields = np.ones((10, 3, 2, 3), dtype=np.float32)
        water_mask = np.ones((2, 3), dtype=bool)
        coordinates = np.zeros((2, 3), dtype=np.float32)

        samples = _extract_partition_samples(
            times,
            fields,
            water_mask,
            coordinates,
            coordinates,
        )

        self.assertEqual(len(samples), 4)
        self.assertEqual(samples[0].inputs.shape, (6, 2, 3))
        self.assertEqual(
            samples[0].target.shape,
            (FORECAST_STEPS, 3, 2, 3),
        )
        self.assertEqual(
            samples[0].sample_id,
            "real_2020-01-01",
        )

    def test_land_inputs_are_zero_and_land_targets_remain_nan(self):
        times = np.datetime64("2020-01-01") + np.arange(8) * np.timedelta64(1, "D")

        water_mask = np.array(
            [
                [True, True],
                [True, False],
            ]
        )

        fields = np.ones((8, 3, 2, 2), dtype=np.float32)
        fields[:, :, 1, 1] = np.nan

        coordinates = np.zeros((2, 2), dtype=np.float32)

        samples = _extract_partition_samples(
            times,
            fields,
            water_mask,
            coordinates,
            coordinates,
        )

        sample = samples[0]

        # The first three channels contain the initial ocean state.
        self.assertTrue(np.isfinite(sample.inputs[:3]).all())
        self.assertEqual(sample.inputs[0, 1, 1], 0.0)
        self.assertEqual(sample.inputs[1, 1, 1], 0.0)
        self.assertEqual(sample.inputs[2, 1, 1], 0.0)

        # The fourth channel identifies land and water.
        self.assertEqual(sample.inputs[3, 1, 1], 0.0)
        self.assertEqual(sample.inputs[3, 0, 0], 1.0)

        # Target values at land cells remain NaN.
        self.assertTrue(np.isnan(sample.target[:, :, 1, 1]).all())
        self.assertTrue(np.isfinite(sample.target[:, :, 0, 0]).all())

    def test_manifest_accepts_complete_monthly_file_list(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "manifest.csv"

            records = []

            for year in range(2020, 2026):
                for month in range(1, 13):
                    month_label = f"{year}-{month:02d}"
                    filename = f"{month_label}.nc"
                    (root / filename).touch()

                    if month == 12:
                        end = f"{year + 1}-01-01T00:00:00"
                    else:
                        end = f"{year}-{month + 1:02d}-01T00:00:00"

                    records.append(
                        {
                            "month": month_label,
                            "start": f"{month_label}-01T00:00:00",
                            "end": end,
                            "time_steps": str(monthrange(year, month)[1]),
                            "source_file": filename,
                            "status": "success",
                        }
                    )

            with manifest_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=records[0].keys(),
                )
                writer.writeheader()
                writer.writerows(records)

            monthly_files = _read_manifest(manifest_path, root)

            self.assertEqual(len(monthly_files), 72)
            self.assertEqual(monthly_files[0][0], "2020-01")
            self.assertEqual(monthly_files[-1][0], "2025-12")

    def test_manifest_rejects_unsuccessful_month(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "manifest.csv"

            records = []

            for year in range(2020, 2026):
                for month in range(1, 13):
                    month_label = f"{year}-{month:02d}"
                    filename = f"{month_label}.nc"
                    (root / filename).touch()

                    if month == 12:
                        end = f"{year + 1}-01-01T00:00:00"
                    else:
                        end = f"{year}-{month + 1:02d}-01T00:00:00"

                    records.append(
                        {
                            "month": month_label,
                            "start": f"{month_label}-01T00:00:00",
                            "end": end,
                            "time_steps": str(monthrange(year, month)[1]),
                            "source_file": filename,
                            "status": "success",
                        }
                    )

            records[0]["status"] = "failed"

            with manifest_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=records[0].keys(),
                )
                writer.writeheader()
                writer.writerows(records)

            with self.assertRaisesRegex(ValueError, "not marked successful"):
                _read_manifest(manifest_path, root)

    def test_load_month_reads_copernicus_variables(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "january.nc"

            times = np.datetime64("2020-01-01") + np.arange(31) * np.timedelta64(1, "D")

            dataset = xr.Dataset(
                {
                    "zos": (
                        ("time", "latitude", "longitude"),
                        np.ones((31, 2, 3), dtype=np.float32),
                    ),
                    "uo": (
                        ("time", "depth", "latitude", "longitude"),
                        np.ones((31, 1, 2, 3), dtype=np.float32),
                    ),
                    "vo": (
                        ("time", "depth", "latitude", "longitude"),
                        np.ones((31, 1, 2, 3), dtype=np.float32),
                    ),
                },
                coords={
                    "time": times,
                    "depth": [0.49402499198913574],
                    "latitude": [-4.5, -4.4],
                    "longitude": [40.0, 40.1, 40.2],
                },
            )

            dataset.to_netcdf(path, engine="scipy")

            loaded_times, longitude, latitude, fields = _load_month(
                path,
                "2020-01",
            )

            self.assertEqual(len(loaded_times), 31)
            self.assertEqual(longitude.shape, (3,))
            self.assertEqual(latitude.shape, (2,))
            self.assertEqual(fields.shape, (31, 3, 2, 3))
            self.assertEqual(fields.dtype, np.float32)
