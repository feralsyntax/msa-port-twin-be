import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import xarray as xr
from django.test import SimpleTestCase

from port_twin_app.services.marine_data.ingestion import (
    ingest_marine_data,
)
from port_twin_app.services.marine_data.normalization import (
    normalize_copernicus_dataset,
    normalize_gebco_bathymetry,
)
from port_twin_app.services.marine_data.sources.base import (
    MarineDataRequest,
    MarineDataSource,
)
from port_twin_app.services.marine_data.sources.copernicus import (
    CopernicusMarineSource,
)
from port_twin_app.services.marine_data.validation import (
    validate_bathymetry_dataset,
    validate_marine_dataset,
)


def create_test_dataset():
    """Create a small Copernicus-like dataset."""

    shape = (3, 2, 2)

    zos = np.full(shape, 0.48)
    uo = np.full((3, 1, 2, 2), 0.12)
    vo = np.full((3, 1, 2, 2), 0.24)

    zos[:, 0, 0] = np.nan
    uo[:, :, 0, 0] = np.nan
    vo[:, :, 0, 0] = np.nan

    return xr.Dataset(
        {
            "zos": (
                ("time", "latitude", "longitude"),
                zos,
                {"units": "m"},
            ),
            "uo": (
                ("time", "depth", "latitude", "longitude"),
                uo,
                {"units": "m s-1"},
            ),
            "vo": (
                ("time", "depth", "latitude", "longitude"),
                vo,
                {"units": "m s-1"},
            ),
        },
        coords={
            "time": np.array(
                [
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                ],
                dtype="datetime64[ns]",
            ),
            "depth": [0.494],
            "latitude": [-4.1, -4.0],
            "longitude": [39.6, 39.7],
        },
        attrs={
            "provider": "Copernicus Marine",
            "source_dataset_id": "test-dataset",
            "source_dataset_version": "202311",
            "source_version": "GLORYS12V1",
            "source_file_path": "test.nc",
        },
    )


def create_test_gebco_dataset():
    """Create a small GEBCO-like bathymetry dataset."""

    return xr.Dataset(
        {
            "elevation_m": (
                ("latitude", "longitude"),
                np.array(
                    [
                        [-100.0, -25.0, 0.0],
                        [10.0, np.nan, -5.0],
                    ],
                    dtype=np.float32,
                ),
                {
                    "units": "m",
                    "standard_name": "height_above_mean_sea_level",
                },
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
            "vertical_reference": "height_above_mean_sea_level",
            "vertical_crs": "EPSG:5831",
            "elevation_positive": "up",
        },
    )


class StubMarineDataSource(MarineDataSource):
    def __init__(self, file_path="test.nc"):
        self.file_path = str(file_path)

    def fetch(self, request):
        dataset = create_test_dataset()
        dataset.attrs["source_file_path"] = self.file_path
        return dataset


class MarineDataRequestTests(SimpleTestCase):
    def create_request(self, **overrides):
        """Create a valid request, allowing individual fields to be overridden."""

        parameters = {
            "west": 39.5,
            "south": -4.3,
            "east": 40.0,
            "north": -3.7,
            "start_time": datetime(2020, 1, 1, tzinfo=UTC),
            "end_time": datetime(2020, 1, 3, tzinfo=UTC),
            "variables": ("zos", "uo", "vo"),
            "minimum_depth": 0,
            "maximum_depth": 1,
        }

        parameters.update(overrides)

        return MarineDataRequest(**parameters)

    def test_accepts_valid_request(self):
        request = self.create_request()

        self.assertEqual(request.variables, ("zos", "uo", "vo"))

    def test_rejects_invalid_longitude(self):
        with self.assertRaisesRegex(ValueError, "longitude"):
            self.create_request(west=-181)

    def test_rejects_invalid_latitude(self):
        with self.assertRaisesRegex(ValueError, "latitude"):
            self.create_request(north=91)

    def test_rejects_reversed_longitude_bounds(self):
        with self.assertRaisesRegex(ValueError, "Western longitude"):
            self.create_request(west=40.0, east=39.5)

    def test_rejects_reversed_latitude_bounds(self):
        with self.assertRaisesRegex(ValueError, "Southern latitude"):
            self.create_request(south=-3.7, north=-4.3)

    def test_rejects_reversed_time_range(self):
        with self.assertRaisesRegex(ValueError, "Start time"):
            self.create_request(
                start_time=datetime(2020, 1, 3, tzinfo=UTC),
                end_time=datetime(2020, 1, 1, tzinfo=UTC),
            )

    def test_rejects_inconsistent_timezone_awareness(self):
        with self.assertRaisesRegex(ValueError, "timezone awareness"):
            self.create_request(
                end_time=datetime(2020, 1, 3),
            )

    def test_rejects_empty_variables(self):
        with self.assertRaisesRegex(ValueError, "variable"):
            self.create_request(variables=())

    def test_rejects_blank_variable_names(self):
        with self.assertRaisesRegex(ValueError, "variable"):
            self.create_request(variables=("zos", " "))

    def test_rejects_non_string_variable_names(self):
        with self.assertRaisesRegex(ValueError, "variable"):
            self.create_request(variables=("zos", 123))

    def test_rejects_negative_minimum_depth(self):
        with self.assertRaisesRegex(ValueError, "Minimum depth"):
            self.create_request(minimum_depth=-1)

    def test_rejects_negative_maximum_depth(self):
        with self.assertRaisesRegex(ValueError, "Maximum depth"):
            self.create_request(maximum_depth=-1)

    def test_rejects_reversed_depth_range(self):
        with self.assertRaisesRegex(ValueError, "Minimum depth"):
            self.create_request(
                minimum_depth=10,
                maximum_depth=5,
            )

    def test_rejects_non_finite_coordinates(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            self.create_request(west=float("nan"))

    def test_rejects_non_finite_depth(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            self.create_request(maximum_depth=float("inf"))


class MarineDataNormalizationTests(SimpleTestCase):
    def test_normalizes_copernicus_variables(self):
        normalized = normalize_copernicus_dataset(create_test_dataset())

        self.assertIn("sea_surface_height_m", normalized)
        self.assertIn("current_u_ms", normalized)
        self.assertIn("current_v_ms", normalized)

        self.assertNotIn("depth", normalized.dims)

        self.assertEqual(
            normalized["current_u_ms"].dims,
            ("time", "latitude", "longitude"),
        )

    def test_preserves_missing_values(self):
        normalized = normalize_copernicus_dataset(create_test_dataset())

        self.assertTrue(
            np.isnan(normalized["sea_surface_height_m"].values[:, 0, 0]).all()
        )

    def test_rejects_missing_variables(self):
        dataset = create_test_dataset().drop_vars("uo")

        with self.assertRaises(ValueError):
            normalize_copernicus_dataset(dataset)

    def test_rejects_unsupported_sea_surface_height_units(self):
        dataset = create_test_dataset()
        dataset["zos"].attrs["units"] = "cm"

        with self.assertRaisesRegex(ValueError, "unsupported units"):
            normalize_copernicus_dataset(dataset)

    def test_rejects_unsupported_eastward_current_units(self):
        dataset = create_test_dataset()
        dataset["uo"].attrs["units"] = "knots"

        with self.assertRaisesRegex(ValueError, "unsupported units"):
            normalize_copernicus_dataset(dataset)

    def test_rejects_unsupported_northward_current_units(self):
        dataset = create_test_dataset()
        dataset["vo"].attrs["units"] = "knots"

        with self.assertRaisesRegex(ValueError, "unsupported units"):
            normalize_copernicus_dataset(dataset)

    def test_normalizes_gebco_elevation_to_positive_downward_depth(self):
        normalized = normalize_gebco_bathymetry(create_test_gebco_dataset())

        expected = np.array(
            [
                [100.0, 25.0, np.nan],
                [np.nan, np.nan, 5.0],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(
            normalized["bathymetry_m"].values,
            expected,
            equal_nan=True,
        )

    def test_preserves_original_gebco_elevation(self):
        dataset = create_test_gebco_dataset()
        normalized = normalize_gebco_bathymetry(dataset)

        self.assertIn("elevation_m", normalized)

        np.testing.assert_array_equal(
            normalized["elevation_m"].values,
            dataset["elevation_m"].values,
        )

    def test_preserves_missing_values_and_masks_land(self):
        normalized = normalize_gebco_bathymetry(create_test_gebco_dataset())

        bathymetry = normalized["bathymetry_m"].values

        self.assertTrue(np.isnan(bathymetry[0, 2]))
        self.assertTrue(np.isnan(bathymetry[1, 0]))
        self.assertTrue(np.isnan(bathymetry[1, 1]))

    def test_sets_bathymetry_metadata(self):
        normalized = normalize_gebco_bathymetry(create_test_gebco_dataset())

        self.assertEqual(
            normalized["bathymetry_m"].attrs["units"],
            "m",
        )
        self.assertEqual(
            normalized["bathymetry_m"].attrs["positive"],
            "down",
        )
        self.assertEqual(
            normalized.attrs["normalization_source"],
            "GEBCO",
        )
        self.assertEqual(
            normalized.attrs["bathymetry_positive"],
            "down",
        )

    def test_rejects_missing_elevation_variable(self):
        dataset = create_test_gebco_dataset().drop_vars("elevation_m")

        with self.assertRaisesRegex(ValueError, "missing the elevation_m"):
            normalize_gebco_bathymetry(dataset)

    def test_rejects_unsupported_elevation_units(self):
        dataset = create_test_gebco_dataset()
        dataset["elevation_m"].attrs["units"] = "feet"

        with self.assertRaisesRegex(ValueError, "must be in metres"):
            normalize_gebco_bathymetry(dataset)

    def test_rejects_dataset_without_valid_underwater_values(self):
        dataset = create_test_gebco_dataset()
        dataset["elevation_m"].values[:] = 10.0

        with self.assertRaisesRegex(ValueError, "no valid underwater"):
            normalize_gebco_bathymetry(dataset)


class MarineDataValidationTests(SimpleTestCase):
    def test_accepts_valid_normalized_dataset(self):
        normalized = normalize_copernicus_dataset(create_test_dataset())

        validate_marine_dataset(normalized)

    def test_rejects_inconsistent_missing_masks(self):
        normalized = normalize_copernicus_dataset(create_test_dataset())

        normalized["current_u_ms"].values[:, 0, 0] = 0.1

        with self.assertRaises(ValueError):
            validate_marine_dataset(normalized)


class MarineDataIngestionTests(SimpleTestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        self.source_file = Path(self.temp_dir.name) / "test.nc"

        self.request = MarineDataRequest(
            west=39.5,
            south=-4.3,
            east=40.0,
            north=-3.7,
            start_time=datetime(2020, 1, 1, tzinfo=UTC),
            end_time=datetime(2020, 1, 3, tzinfo=UTC),
            variables=("zos", "uo", "vo"),
            minimum_depth=0,
            maximum_depth=1,
        )

    def test_ingests_and_records_provenance(self):
        result = ingest_marine_data(
            StubMarineDataSource(self.source_file),
            self.request,
            license_name="Test license",
        )

        self.assertEqual(
            result.provenance.provider,
            "Copernicus Marine",
        )

        self.assertEqual(
            result.provenance.dataset_version,
            "202311",
        )

        self.assertIn(
            "sea_surface_height_m",
            result.dataset,
        )

        self.assertEqual(
            result.provenance.to_dict()["license_name"],
            "Test license",
        )

    def test_rejects_empty_license(self):
        with self.assertRaises(ValueError):
            ingest_marine_data(
                StubMarineDataSource(self.source_file),
                self.request,
                license_name=" ",
            )

    def test_persists_provenance_metadata(self):
        with TemporaryDirectory() as temp_dir:
            source_file = Path(temp_dir) / "test.nc"
            source_file.touch()

            result = ingest_marine_data(
                StubMarineDataSource(source_file),
                self.request,
                license_name="Test license",
            )

            provenance_path = Path(result.provenance_file_path)

            self.assertTrue(provenance_path.is_file())

            metadata = json.loads(provenance_path.read_text(encoding="utf-8"))

            self.assertEqual(
                metadata["provider"],
                "Copernicus Marine",
            )

            self.assertEqual(
                metadata["dataset_version"],
                "202311",
            )


class CopernicusMarineSourceTests(SimpleTestCase):
    def setUp(self):
        self.source = CopernicusMarineSource()

    @patch(
        "port_twin_app.services.marine_data.sources.copernicus.copernicusmarine.describe"
    )
    def test_resolves_default_dataset_version(self, mock_describe):
        mock_describe.return_value = SimpleNamespace(
            products=[
                SimpleNamespace(
                    datasets=[
                        SimpleNamespace(
                            dataset_id=self.source.dataset_id,
                            versions=[SimpleNamespace(label="202311")],
                        )
                    ]
                )
            ]
        )

        version = self.source._resolve_dataset_version()

        self.assertEqual(version, "202311")

        mock_describe.assert_called_once_with(
            dataset_id=self.source.dataset_id,
            show_all_versions=False,
            disable_progress_bar=True,
            raise_on_error=True,
        )

    @patch(
        "port_twin_app.services.marine_data.sources.copernicus.copernicusmarine.describe"
    )
    def test_raises_error_when_dataset_version_is_missing(self, mock_describe):
        mock_describe.return_value = SimpleNamespace(
            products=[
                SimpleNamespace(
                    datasets=[
                        SimpleNamespace(
                            dataset_id=self.source.dataset_id,
                            versions=[],
                        )
                    ]
                )
            ]
        )

        with self.assertRaises(ValueError):
            self.source._resolve_dataset_version()

    @patch(
        "port_twin_app.services.marine_data.sources.copernicus.copernicusmarine.describe"
    )
    def test_raises_error_when_dataset_is_not_found(self, mock_describe):
        mock_describe.return_value = SimpleNamespace(products=[])

        with self.assertRaises(ValueError):
            self.source._resolve_dataset_version()

    @patch(
        "port_twin_app.services.marine_data.sources.copernicus.copernicusmarine.subset"
    )
    @patch(
        "port_twin_app.services.marine_data.sources.copernicus.copernicusmarine.describe"
    )
    def test_fetch_uses_resolved_dataset_version(
        self,
        mock_describe,
        mock_subset,
    ):
        mock_describe.return_value = SimpleNamespace(
            products=[
                SimpleNamespace(
                    datasets=[
                        SimpleNamespace(
                            dataset_id=self.source.dataset_id,
                            versions=[SimpleNamespace(label="202311")],
                        )
                    ]
                )
            ]
        )

        with TemporaryDirectory() as temp_dir:
            source = CopernicusMarineSource(output_directory=temp_dir)

            def create_downloaded_file(**kwargs):
                dataset = xr.Dataset(
                    {
                        "zos": (
                            ("time", "latitude", "longitude"),
                            np.array([[[0.48]]]),
                            {"units": "m"},
                        )
                    },
                    coords={
                        "time": [np.datetime64("2020-01-01")],
                        "latitude": [-4.0],
                        "longitude": [39.7],
                    },
                )

                file_path = (
                    Path(kwargs["output_directory"]) / f"{kwargs['output_filename']}.nc"
                )

                dataset.to_netcdf(file_path)

            mock_subset.side_effect = create_downloaded_file

            request = MarineDataRequest(
                west=39.5,
                south=-4.3,
                east=40.0,
                north=-3.7,
                start_time=datetime(2020, 1, 1),
                end_time=datetime(2020, 1, 1),
                variables=("zos",),
            )

            result = source.fetch(request)

            self.assertEqual(
                mock_subset.call_args.kwargs["dataset_version"],
                "202311",
            )

            self.assertEqual(
                result.attrs["source_dataset_version"],
                "202311",
            )

            mock_describe.assert_called_once()


class BathymetryValidationTests(SimpleTestCase):
    def create_valid_dataset(self):
        """Create a normalized GEBCO bathymetry dataset."""

        return normalize_gebco_bathymetry(create_test_gebco_dataset())

    def test_accepts_valid_bathymetry_dataset(self):
        dataset = self.create_valid_dataset()

        validate_bathymetry_dataset(dataset)

    def test_accepts_missing_values(self):
        dataset = self.create_valid_dataset()

        validate_bathymetry_dataset(dataset)

    def test_rejects_missing_bathymetry_variable(self):
        dataset = self.create_valid_dataset().drop_vars("bathymetry_m")

        with self.assertRaisesRegex(ValueError, "bathymetry_m"):
            validate_bathymetry_dataset(dataset)

    def test_rejects_missing_coordinates(self):
        dataset = self.create_valid_dataset().drop_vars("longitude")

        with self.assertRaisesRegex(ValueError, "longitude"):
            validate_bathymetry_dataset(dataset)

    def test_rejects_insufficient_spatial_grid(self):
        dataset = self.create_valid_dataset().isel(latitude=[0])

        with self.assertRaisesRegex(ValueError, "at least 2 points"):
            validate_bathymetry_dataset(dataset)

    def test_rejects_non_finite_coordinates(self):
        dataset = self.create_valid_dataset()
        dataset = dataset.assign_coords(latitude=[-4.1, np.nan])

        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_bathymetry_dataset(dataset)

    def test_rejects_non_increasing_coordinates(self):
        dataset = self.create_valid_dataset()
        dataset = dataset.assign_coords(latitude=[-4.0, -4.1])

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            validate_bathymetry_dataset(dataset)

    def test_rejects_invalid_latitude(self):
        dataset = self.create_valid_dataset()
        dataset = dataset.assign_coords(latitude=[-91.0, -90.0])

        with self.assertRaisesRegex(ValueError, "Latitude"):
            validate_bathymetry_dataset(dataset)

    def test_rejects_invalid_longitude(self):
        dataset = self.create_valid_dataset()
        dataset = dataset.assign_coords(longitude=[39.6, 361.0, 362.0])

        with self.assertRaisesRegex(ValueError, "Longitude"):
            validate_bathymetry_dataset(dataset)

    def test_rejects_unexpected_bathymetry_dimensions(self):
        dataset = self.create_valid_dataset()
        dataset["bathymetry_m"] = dataset["bathymetry_m"].transpose(
            "longitude",
            "latitude",
        )

        with self.assertRaisesRegex(ValueError, "unexpected dimensions"):
            validate_bathymetry_dataset(dataset)

    def test_rejects_unsupported_bathymetry_units(self):
        dataset = self.create_valid_dataset()
        dataset["bathymetry_m"].attrs["units"] = "feet"

        with self.assertRaisesRegex(ValueError, "metres"):
            validate_bathymetry_dataset(dataset)

    def test_rejects_wrong_positive_direction(self):
        dataset = self.create_valid_dataset()
        dataset["bathymetry_m"].attrs["positive"] = "up"

        with self.assertRaisesRegex(ValueError, "positive-down"):
            validate_bathymetry_dataset(dataset)

    def test_rejects_infinite_bathymetry_values(self):
        dataset = self.create_valid_dataset()
        dataset["bathymetry_m"].values[0, 0] = np.inf

        with self.assertRaisesRegex(ValueError, "infinite"):
            validate_bathymetry_dataset(dataset)

    def test_rejects_non_positive_bathymetry_depths(self):
        dataset = self.create_valid_dataset()
        dataset["bathymetry_m"].values[0, 0] = 0.0

        with self.assertRaisesRegex(ValueError, "greater than zero"):
            validate_bathymetry_dataset(dataset)

    def test_rejects_dataset_without_valid_bathymetry(self):
        dataset = self.create_valid_dataset()
        dataset["bathymetry_m"][:] = np.nan

        with self.assertRaisesRegex(ValueError, "no valid underwater"):
            validate_bathymetry_dataset(dataset)
