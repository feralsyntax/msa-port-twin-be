from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import xarray as xr
from django.test import SimpleTestCase

from port_twin_app.services.marine_data.ingestion import (
    ingest_marine_data,
)
from port_twin_app.services.marine_data.normalization import (
    normalize_copernicus_dataset,
)
from port_twin_app.services.marine_data.sources.base import (
    MarineDataRequest,
    MarineDataSource,
)
from port_twin_app.services.marine_data.validation import (
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


class StubMarineDataSource(MarineDataSource):
    def __init__(self, file_path="test.nc"):
        self.file_path = str(file_path)

    def fetch(self, request):
        dataset = create_test_dataset()
        dataset.attrs["source_file_path"] = self.file_path
        return dataset


class MarineDataNormalizationTests(SimpleTestCase):
    def test_normalizes_copernicus_variables(self):
        normalized = normalize_copernicus_dataset(create_test_dataset())

        self.assertIn("water_level_m", normalized)
        self.assertIn("current_u_ms", normalized)
        self.assertIn("current_v_ms", normalized)

        self.assertNotIn("depth", normalized.dims)

        self.assertEqual(
            normalized["current_u_ms"].dims,
            ("time", "latitude", "longitude"),
        )

    def test_preserves_missing_values(self):
        normalized = normalize_copernicus_dataset(create_test_dataset())

        self.assertTrue(np.isnan(normalized["water_level_m"].values[:, 0, 0]).all())

    def test_rejects_missing_variables(self):
        dataset = create_test_dataset().drop_vars("uo")

        with self.assertRaises(ValueError):
            normalize_copernicus_dataset(dataset)


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
            "water_level_m",
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

            import json

            metadata = json.loads(provenance_path.read_text(encoding="utf-8"))

            self.assertEqual(
                metadata["provider"],
                "Copernicus Marine",
            )

            self.assertEqual(
                metadata["dataset_version"],
                "202311",
            )
