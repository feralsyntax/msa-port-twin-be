import numpy as np
import xarray as xr
from django.test import SimpleTestCase

from port_twin_app.services.marine_ai.preparation import (
    prepare_marine_dataset,
)


def create_marine_dataset(time_count=100):
    times = np.datetime64("2026-10-01T00:00:00") + np.arange(
        time_count
    ) * np.timedelta64(1, "h")

    shape = (time_count, 2, 3)

    return xr.Dataset(
        {
            "sea_surface_height_m": (
                ("time", "latitude", "longitude"),
                np.full(shape, 0.48),
            ),
            "current_u_ms": (
                ("time", "latitude", "longitude"),
                np.full(shape, 0.12),
            ),
            "current_v_ms": (
                ("time", "latitude", "longitude"),
                np.full(shape, 0.24),
            ),
        },
        coords={
            "time": times,
            "latitude": [-4.1, -4.0],
            "longitude": [39.6, 39.7, 39.8],
        },
        attrs={
            "provider": "Copernicus Marine",
            "dataset_id": "test-dataset",
            "dataset_version": "202406",
            "vertical_reference": "geoid",
            "license_name": "Copernicus Marine Service Licence",
        },
    )


def create_bathymetry_dataset():
    return xr.Dataset(
        {
            "bathymetry_m": (
                ("latitude", "longitude"),
                np.array(
                    [
                        [100.0, 110.0, 120.0],
                        [130.0, 140.0, 150.0],
                    ]
                ),
            ),
            "tid": (
                ("latitude", "longitude"),
                np.array(
                    [
                        [10, 11, 40],
                        [40, 44, 16],
                    ],
                    dtype=np.int16,
                ),
            ),
        },
        coords={
            "latitude": [-4.1, -4.0],
            "longitude": [39.6, 39.7, 39.8],
        },
        attrs={
            "provider": "GEBCO",
            "source_dataset_id": "GEBCO",
            "source_dataset_version": "GEBCO_2026",
            "source_version": "GEBCO_2026",
            "vertical_reference": "height_above_mean_sea_level",
            "vertical_crs": "EPSG:5831",
            "license": "GEBCO public domain",
            "tid_source_file_path": "gebco_tid.nc",
        },
    )


class MarineDatasetPreparationTests(SimpleTestCase):
    def setUp(self):
        self.marine = create_marine_dataset()
        self.bathymetry = create_bathymetry_dataset()

    def prepare(self, marine=None, bathymetry=None):
        return prepare_marine_dataset(
            marine if marine is not None else self.marine,
            bathymetry if bathymetry is not None else self.bathymetry,
        )

    def test_valid_datasets_generate_six_hour_samples(self):
        result = self.prepare()

        self.assertEqual(result.metadata["generated_samples"], 94)
        self.assertEqual(result.metadata["purged_samples"], 12)
        self.assertEqual(result.split.total_samples, 82)

        sample = result.split.train[0]

        self.assertEqual(sample.request.forecast.steps, 6)
        self.assertEqual(sample.request.forecast.step_seconds, 3600)
        self.assertEqual(len(sample.target.water_level_m), 6)
        self.assertEqual(len(sample.target.current_u_ms), 6)
        self.assertEqual(len(sample.target.current_v_ms), 6)

    def test_samples_use_initial_observation_and_next_six_hours(self):
        result = self.prepare()

        sample = result.split.train[0]

        self.assertEqual(
            sample.request.initial_water_level_m,
            [[0.48] * 3] * 2,
        )
        self.assertEqual(
            sample.target.water_level_m,
            [[[0.48] * 3] * 2 for _ in range(6)],
        )

    def test_bathymetry_is_included_in_model_inputs(self):
        result = self.prepare()

        self.assertEqual(
            result.split.train[0].request.bathymetry_m,
            [[100.0, 110.0, 120.0], [130.0, 140.0, 150.0]],
        )

    def test_dataset_is_split_into_nonempty_partitions(self):
        result = self.prepare()

        self.assertTrue(result.split.train)
        self.assertTrue(result.split.validation)
        self.assertTrue(result.split.test)

        self.assertEqual(
            result.metadata["train_samples"]
            + result.metadata["validation_samples"]
            + result.metadata["test_samples"],
            result.split.total_samples,
        )

    def test_samples_are_chronologically_separated(self):
        result = self.prepare()

        self.assertTrue(
            all(
                sample.forecast_end < result.split.validation[0].forecast_start
                for sample in result.split.train
            )
        )
        self.assertTrue(
            all(
                sample.forecast_end < result.split.test[0].forecast_start
                for sample in result.split.validation
            )
        )

    def test_tid_classifications_are_preserved_as_counts(self):
        result = self.prepare()

        self.assertEqual(
            result.tid_summary,
            {
                10: 1,
                11: 1,
                16: 1,
                40: 2,
                44: 1,
            },
        )

    def test_source_and_vertical_reference_metadata_are_preserved(self):
        result = self.prepare()

        self.assertEqual(
            result.metadata["marine_dataset_version"],
            "202406",
        )
        self.assertEqual(
            result.metadata["bathymetry_dataset_version"],
            "GEBCO_2026",
        )
        self.assertEqual(
            result.metadata["marine_vertical_reference"],
            "geoid",
        )
        self.assertEqual(
            result.metadata["bathymetry_vertical_reference"],
            "height_above_mean_sea_level",
        )
        self.assertIsNotNone(result.metadata["vertical_reference_warning"])

    def test_missing_marine_values_are_rejected(self):
        marine = self.marine.copy(deep=True)
        marine["current_u_ms"][0, 0, 0] = np.nan

        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.prepare(marine=marine)

    def test_missing_bathymetry_values_are_rejected(self):
        bathymetry = self.bathymetry.copy(deep=True)
        bathymetry["bathymetry_m"][0, 0] = np.nan

        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.prepare(bathymetry=bathymetry)

    def test_non_positive_bathymetry_is_rejected(self):
        bathymetry = self.bathymetry.copy(deep=True)
        bathymetry["bathymetry_m"][0, 0] = 0.0

        with self.assertRaisesRegex(ValueError, "positive"):
            self.prepare(bathymetry=bathymetry)

    def test_mismatched_latitude_coordinates_are_rejected(self):
        bathymetry = self.bathymetry.assign_coords(latitude=[-4.11, -4.0])

        with self.assertRaisesRegex(ValueError, "latitude"):
            self.prepare(bathymetry=bathymetry)

    def test_mismatched_longitude_coordinates_are_rejected(self):
        bathymetry = self.bathymetry.assign_coords(longitude=[39.6, 39.71, 39.8])

        with self.assertRaisesRegex(ValueError, "longitude"):
            self.prepare(bathymetry=bathymetry)

    def test_non_hourly_observations_are_rejected(self):
        marine = self.marine.copy(deep=True)
        times = marine.time.values.copy()
        times[10:] += np.timedelta64(1, "h")
        marine = marine.assign_coords(time=times)

        with self.assertRaisesRegex(ValueError, "hourly"):
            self.prepare(marine=marine)

    def test_missing_tid_is_rejected(self):
        bathymetry = self.bathymetry.drop_vars("tid")

        with self.assertRaisesRegex(ValueError, "TID"):
            self.prepare(bathymetry=bathymetry)

    def test_non_integer_tid_is_rejected(self):
        bathymetry = self.bathymetry.copy(deep=True)
        bathymetry["tid"] = bathymetry["tid"].astype(float)
        bathymetry["tid"][0, 0] = 10.5

        with self.assertRaisesRegex(ValueError, "integer"):
            self.prepare(bathymetry=bathymetry)

    def test_missing_tid_values_are_rejected(self):
        bathymetry = self.bathymetry.copy(deep=True)
        bathymetry["tid"] = bathymetry["tid"].astype(float)
        bathymetry["tid"][0, 0] = np.nan

        with self.assertRaisesRegex(ValueError, "TID"):
            self.prepare(bathymetry=bathymetry)

    def test_non_dataset_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "xarray Dataset"):
            self.prepare(marine=[])

    def test_insufficient_time_steps_are_rejected(self):
        marine = create_marine_dataset(time_count=6)

        with self.assertRaisesRegex(ValueError, "hourly timestamps"):
            self.prepare(marine=marine)
