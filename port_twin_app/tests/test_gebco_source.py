from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import xarray as xr
from django.test import SimpleTestCase

from port_twin_app.services.marine_data.sources.bathymetry import (
    BathymetryRequest,
)
from port_twin_app.services.marine_data.sources.gebco import (
    GEBCOBathymetrySource,
)


class GEBCOBathymetrySourceTests(SimpleTestCase):
    def setUp(self):
        self.temp_directory = TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)

        self.file_path = Path(self.temp_directory.name) / "gebco_test.nc"
        self.tid_file_path = Path(self.temp_directory.name) / "gebco_tid_test.nc"

        longitude = np.array([39.0, 39.5, 40.0, 40.5, 41.0])
        latitude = np.array([-5.0, -4.5, -4.0, -3.5, -3.0])

        elevation = np.array(
            [
                [-100, -110, -120, -130, -140],
                [-90, -100, -110, -120, -130],
                [-80, -90, -100, -110, -120],
                [-70, -80, -90, -100, -110],
                [-60, -70, -80, -90, -100],
            ],
            dtype=float,
        )

        dataset = xr.Dataset(
            data_vars={
                "elevation": (
                    ("lat", "lon"),
                    elevation,
                    {"units": "m"},
                )
            },
            coords={
                "lon": longitude,
                "lat": latitude,
            },
            attrs={
                "title": "Synthetic GEBCO test dataset",
                "vertical_datum": "Mean Sea Level",
                "source": "GEBCO_2026",
            },
        )

        dataset.to_netcdf(self.file_path, engine="scipy")

        tid = np.array(
            [
                [0, 10, 10, 40, 40],
                [0, 10, 11, 40, 40],
                [0, 14, 16, 40, 44],
                [0, 14, 16, 40, 44],
                [0, 10, 10, 40, 44],
            ],
            dtype=np.int8,
        )

        tid_dataset = xr.Dataset(
            data_vars={
                "tid": (
                    ("lat", "lon"),
                    tid,
                    {
                        "flag_values": np.array(
                            [0, 10, 11, 14, 16, 40, 44],
                            dtype=np.int8,
                        ),
                        "flag_meanings": (
                            "Land Singlebeam Multibeam Lidar Optical "
                            "predicted_altimetry contour_charts"
                        ),
                    },
                )
            },
            coords={
                "lon": longitude,
                "lat": latitude,
            },
        )

        tid_dataset.to_netcdf(self.tid_file_path, engine="scipy")

    def make_request(self, **overrides):
        parameters = {
            "west": 39.5,
            "south": -4.5,
            "east": 40.5,
            "north": -3.5,
        }
        parameters.update(overrides)

        return BathymetryRequest(**parameters)

    def make_source(self, file_path=None, tid_file_path=None):
        return GEBCOBathymetrySource(
            file_path=file_path or self.file_path,
            dataset_version="GEBCO_2026",
            tid_file_path=tid_file_path,
        )

    def test_fetch_returns_requested_geographical_subset(self):
        result = self.make_source().fetch(self.make_request())

        self.assertEqual(result.sizes["longitude"], 3)
        self.assertEqual(result.sizes["latitude"], 3)

        np.testing.assert_array_equal(
            result.longitude.values,
            [39.5, 40.0, 40.5],
        )

        np.testing.assert_array_equal(
            result.latitude.values,
            [-4.5, -4.0, -3.5],
        )

    def test_fetch_normalizes_variable_and_coordinate_names(self):
        result = self.make_source().fetch(self.make_request())

        self.assertIn("elevation_m", result.data_vars)
        self.assertIn("longitude", result.coords)
        self.assertIn("latitude", result.coords)

    def test_fetch_preserves_elevation_values_and_sign(self):
        result = self.make_source().fetch(self.make_request())

        self.assertTrue(np.all(result["elevation_m"].values < 0))

        self.assertEqual(
            result["elevation_m"].attrs["units"],
            "m",
        )

    def test_fetch_preserves_source_metadata(self):
        result = self.make_source().fetch(self.make_request())

        self.assertEqual(result.attrs["provider"], "GEBCO")
        self.assertEqual(result.attrs["source_dataset_id"], "GEBCO")
        self.assertEqual(
            result.attrs["source_dataset_version"],
            "GEBCO_2026",
        )
        self.assertEqual(
            result.attrs["vertical_reference"],
            "Mean Sea Level",
        )
        self.assertEqual(
            result.attrs["source"],
            "GEBCO_2026",
        )

    def test_fetch_rejects_missing_file(self):
        source = self.make_source(Path(self.temp_directory.name) / "missing.nc")

        with self.assertRaises(FileNotFoundError):
            source.fetch(self.make_request())

    def test_fetch_rejects_region_outside_source_grid(self):
        request = self.make_request(west=38.0)

        with self.assertRaises(ValueError):
            self.make_source().fetch(request)

    def test_fetch_rejects_region_with_insufficient_coordinates(self):
        request = self.make_request(
            west=39.5,
            east=39.6,
            south=-4.5,
            north=-3.5,
        )

        with self.assertRaises(ValueError):
            self.make_source().fetch(request)

    def test_fetch_rejects_dataset_without_elevation_variable(self):
        invalid_path = Path(self.temp_directory.name) / "invalid.nc"

        invalid_dataset = xr.Dataset(
            data_vars={
                "temperature": (
                    ("lat", "lon"),
                    np.ones((5, 5)),
                )
            },
            coords={
                "lon": np.linspace(39.0, 41.0, 5),
                "lat": np.linspace(-5.0, -3.0, 5),
            },
        )

        invalid_dataset.to_netcdf(invalid_path, engine="scipy")

        with self.assertRaises(ValueError):
            self.make_source(invalid_path).fetch(self.make_request())

    def test_fetch_rejects_dataset_without_valid_elevations(self):
        invalid_path = Path(self.temp_directory.name) / "empty.nc"

        invalid_dataset = xr.Dataset(
            data_vars={
                "elevation": (
                    ("lat", "lon"),
                    np.full((5, 5), np.nan),
                )
            },
            coords={
                "lon": np.linspace(39.0, 41.0, 5),
                "lat": np.linspace(-5.0, -3.0, 5),
            },
        )

        invalid_dataset.to_netcdf(invalid_path, engine="scipy")

        with self.assertRaises(ValueError):
            self.make_source(invalid_path).fetch(self.make_request())

    def test_fetch_supports_alternative_coordinate_names(self):
        alternative_path = Path(self.temp_directory.name) / "alternative.nc"

        dataset = xr.Dataset(
            data_vars={
                "elev": (
                    ("latitude", "longitude"),
                    np.full((5, 5), -100.0),
                    {"units": "m"},
                )
            },
            coords={
                "longitude": np.linspace(39.0, 41.0, 5),
                "latitude": np.linspace(-5.0, -3.0, 5),
            },
        )

        dataset.to_netcdf(alternative_path, engine="scipy")

        result = self.make_source(alternative_path).fetch(self.make_request())

        self.assertIn("elevation_m", result.data_vars)
        self.assertIn("longitude", result.coords)
        self.assertIn("latitude", result.coords)

    def test_fetch_includes_tid_when_tid_file_is_configured(self):
        result = self.make_source(tid_file_path=self.tid_file_path).fetch(
            self.make_request()
        )

        self.assertIn("tid", result.data_vars)
        self.assertEqual(result["tid"].shape, result["elevation_m"].shape)

        np.testing.assert_array_equal(
            result["tid"].values,
            [
                [10, 11, 40],
                [14, 16, 40],
                [14, 16, 40],
            ],
        )

    def test_fetch_preserves_tid_classification_metadata(self):
        result = self.make_source(tid_file_path=self.tid_file_path).fetch(
            self.make_request()
        )

        self.assertEqual(
            result["tid"].attrs["flag_meanings"],
            (
                "Land Singlebeam Multibeam Lidar Optical "
                "predicted_altimetry contour_charts"
            ),
        )

        np.testing.assert_array_equal(
            result["tid"].attrs["flag_values"],
            [0, 10, 11, 14, 16, 40, 44],
        )

    def test_fetch_preserves_tid_as_integer_values(self):
        result = self.make_source(tid_file_path=self.tid_file_path).fetch(
            self.make_request()
        )

        self.assertTrue(np.issubdtype(result["tid"].dtype, np.integer))

    def test_fetch_records_tid_source_file_path(self):
        result = self.make_source(tid_file_path=self.tid_file_path).fetch(
            self.make_request()
        )

        self.assertEqual(
            result.attrs["tid_source_file_path"],
            str(self.tid_file_path),
        )

    def test_fetch_rejects_missing_tid_file(self):
        missing_tid_path = Path(self.temp_directory.name) / "missing_tid.nc"

        source = self.make_source(tid_file_path=missing_tid_path)

        with self.assertRaises(FileNotFoundError):
            source.fetch(self.make_request())

    def test_fetch_rejects_tid_grid_with_mismatched_coordinates(self):
        mismatched_tid_path = Path(self.temp_directory.name) / "mismatched_tid.nc"

        dataset = xr.Dataset(
            data_vars={
                "tid": (
                    ("lat", "lon"),
                    np.ones((5, 5), dtype=np.int8),
                )
            },
            coords={
                "lon": np.array([39.0, 39.5, 40.0, 40.5, 41.1]),
                "lat": np.array([-5.0, -4.5, -4.0, -3.5, -3.0]),
            },
        )

        dataset.to_netcdf(mismatched_tid_path, engine="scipy")

        source = self.make_source(tid_file_path=mismatched_tid_path)

        with self.assertRaisesRegex(
            ValueError,
            "longitude coordinates do not match",
        ):
            source.fetch(self.make_request())

    def test_fetch_rejects_tid_dataset_without_tid_variable(self):
        invalid_tid_path = Path(self.temp_directory.name) / "invalid_tid.nc"

        dataset = xr.Dataset(
            data_vars={
                "classification": (
                    ("lat", "lon"),
                    np.ones((5, 5), dtype=np.int8),
                )
            },
            coords={
                "lon": np.linspace(39.0, 41.0, 5),
                "lat": np.linspace(-5.0, -3.0, 5),
            },
        )

        dataset.to_netcdf(invalid_tid_path, engine="scipy")

        source = self.make_source(tid_file_path=invalid_tid_path)

        with self.assertRaisesRegex(ValueError, "TID variable"):
            source.fetch(self.make_request())
