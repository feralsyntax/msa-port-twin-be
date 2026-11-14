"""Tests for assembling validated, dataset-backed marine FNO inputs."""

import json
import tempfile
from datetime import timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from django.test import SimpleTestCase, override_settings
from django.utils import timezone

from port_twin_app.services.marine_ai.input_preparation import (
    GRID_BOUNDS,
    GRID_NX,
    GRID_NY,
    MarineInputUnavailable,
    prepare_marine_prediction_inputs,
)


class MarineInputPreparationTests(SimpleTestCase):
    """Verify marine input assembly using controlled NetCDF datasets."""

    def setUp(self):
        super().setUp()

        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        self.root = Path(self.temp_dir.name)
        self.copernicus_path = self.root / "copernicus.nc"
        self.gebco_path = self.root / "gebco.nc"

        self.start_time = timezone.now().replace(
            minute=0,
            second=0,
            microsecond=0,
        )

        self.start_time_iso = self.start_time.isoformat()

        self._write_copernicus_dataset()
        self._write_gebco_dataset()

        self.settings_override = override_settings(
            MARINE_COPERNICUS_DATASET_PATH=str(self.copernicus_path),
            MARINE_GEBCO_DATASET_PATH=str(self.gebco_path),
            MARINE_DATA_MAX_AGE_HOURS=3,
        )
        self.settings_override.enable()
        self.addCleanup(self.settings_override.disable)

    def _write_provenance(
        self,
        dataset_path,
        *,
        provider,
        dataset_id,
        dataset_version,
        retrieved_at=None,
    ):
        provenance = {
            "provider": provider,
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "source_version": dataset_version,
            "retrieved_at": (retrieved_at or timezone.now()).isoformat(),
            "file_path": str(dataset_path),
            "variables": [],
        }

        provenance_path = dataset_path.with_suffix(".provenance.json")
        provenance_path.write_text(
            json.dumps(provenance),
            encoding="utf-8",
        )

        return provenance

    def _write_copernicus_dataset(self, *, coverage=None):
        if coverage is None:
            latitude = np.linspace(
                GRID_BOUNDS["south"],
                GRID_BOUNDS["north"],
                5,
            )
            longitude = np.linspace(
                GRID_BOUNDS["west"],
                GRID_BOUNDS["east"],
                5,
            )
        else:
            latitude, longitude = coverage

        times = np.array(
            [
                np.datetime64(self.start_time.replace(tzinfo=None), "ns"),
                np.datetime64(
                    (self.start_time + timedelta(hours=1)).replace(tzinfo=None),
                    "ns",
                ),
            ]
        )

        shape = (len(times), len(latitude), len(longitude))

        zos = np.full(shape, 0.25, dtype=np.float64)
        uo = np.full(shape, 0.4, dtype=np.float64)
        vo = np.full(shape, -0.2, dtype=np.float64)

        dataset = xr.Dataset(
            data_vars={
                "zos": (
                    ("time", "latitude", "longitude"),
                    zos,
                    {"units": "m"},
                ),
                "uo": (
                    ("time", "latitude", "longitude"),
                    uo,
                    {"units": "m s-1"},
                ),
                "vo": (
                    ("time", "latitude", "longitude"),
                    vo,
                    {"units": "m s-1"},
                ),
            },
            coords={
                "time": times,
                "latitude": latitude,
                "longitude": longitude,
            },
        )

        dataset.to_netcdf(self.copernicus_path, engine="scipy")

        self._write_provenance(
            self.copernicus_path,
            provider="Copernicus Marine",
            dataset_id="test_copernicus_dataset",
            dataset_version="test-version",
        )

    def _write_gebco_dataset(self, *, coverage=None, elevation=-100.0):
        if coverage is None:
            latitude = np.linspace(
                GRID_BOUNDS["south"],
                GRID_BOUNDS["north"],
                5,
            )
            longitude = np.linspace(
                GRID_BOUNDS["west"],
                GRID_BOUNDS["east"],
                5,
            )
        else:
            latitude, longitude = coverage

        elevation_values = np.full(
            (len(latitude), len(longitude)),
            elevation,
            dtype=np.float64,
        )

        dataset = xr.Dataset(
            data_vars={
                "elevation": (
                    ("lat", "lon"),
                    elevation_values,
                    {"units": "m"},
                ),
            },
            coords={
                "lat": latitude,
                "lon": longitude,
            },
        )

        dataset.to_netcdf(self.gebco_path, engine="scipy")

        self._write_provenance(
            self.gebco_path,
            provider="GEBCO",
            dataset_id="GEBCO",
            dataset_version="GEBCO_TEST",
        )

    def test_assembles_complete_prediction_request(self):
        prepared = prepare_marine_prediction_inputs(
            self.start_time_iso,
        )

        request = prepared.prediction_request

        self.assertEqual(request.grid.nx, GRID_NX)
        self.assertEqual(request.grid.ny, GRID_NY)

        self.assertEqual(len(request.bathymetry_m), GRID_NY)
        self.assertEqual(len(request.initial_water_level_m), GRID_NY)
        self.assertEqual(len(request.initial_current_u_ms), GRID_NY)
        self.assertEqual(len(request.initial_current_v_ms), GRID_NY)

        for field in (
            request.bathymetry_m,
            request.initial_water_level_m,
            request.initial_current_u_ms,
            request.initial_current_v_ms,
        ):
            for row in field:
                self.assertEqual(len(row), GRID_NX)
                self.assertTrue(all(np.isfinite(value) for value in row))

        self.assertTrue(all(value > 0 for row in request.bathymetry_m for value in row))

        self.assertEqual(
            request.initial_water_level_m[0][0],
            0.25,
        )
        self.assertEqual(
            request.initial_current_u_ms[0][0],
            0.4,
        )
        self.assertEqual(
            request.initial_current_v_ms[0][0],
            -0.2,
        )

    def test_returns_grid_forecast_and_source_provenance(self):
        prepared = prepare_marine_prediction_inputs(
            self.start_time_iso,
        )

        self.assertEqual(
            prepared.grid_data["domain"],
            "mombasa_offshore",
        )
        self.assertEqual(prepared.grid_data["nx"], GRID_NX)
        self.assertEqual(prepared.grid_data["ny"], GRID_NY)

        self.assertEqual(
            prepared.forecast_data["start_time"],
            self.start_time_iso,
        )

        marine_source = prepared.input_data["source_data"]["marine"]
        bathymetry_source = prepared.input_data["source_data"]["bathymetry"]

        self.assertEqual(
            marine_source["dataset_id"],
            "test_copernicus_dataset",
        )
        self.assertEqual(
            marine_source["dataset_version"],
            "test-version",
        )
        self.assertEqual(
            marine_source["selected_time"],
            self.start_time_iso,
        )
        self.assertEqual(
            marine_source["file_path"],
            str(self.copernicus_path),
        )

        self.assertEqual(bathymetry_source["dataset_id"], "GEBCO")
        self.assertEqual(
            bathymetry_source["dataset_version"],
            "GEBCO_TEST",
        )
        self.assertEqual(
            bathymetry_source["file_path"],
            str(self.gebco_path),
        )

    def test_rejects_stale_copernicus_dataset(self):
        self._write_provenance(
            self.copernicus_path,
            provider="Copernicus Marine",
            dataset_id="test_copernicus_dataset",
            dataset_version="test-version",
            retrieved_at=timezone.now() - timedelta(hours=4),
        )

        with self.assertRaisesRegex(
            MarineInputUnavailable,
            "Copernicus dataset is stale",
        ):
            prepare_marine_prediction_inputs(self.start_time_iso)

    def test_rejects_missing_copernicus_dataset(self):
        missing_path = self.root / "missing_copernicus.nc"

        with (
            override_settings(
                MARINE_COPERNICUS_DATASET_PATH=str(missing_path),
            ),
            self.assertRaisesRegex(
                MarineInputUnavailable,
                "does not exist",
            ),
        ):
            prepare_marine_prediction_inputs(self.start_time_iso)

    def test_rejects_missing_gebco_dataset(self):
        missing_path = self.root / "missing_gebco.nc"

        with (
            override_settings(
                MARINE_GEBCO_DATASET_PATH=str(missing_path),
            ),
            self.assertRaisesRegex(
                MarineInputUnavailable,
                "does not exist",
            ),
        ):
            prepare_marine_prediction_inputs(self.start_time_iso)

    def test_rejects_missing_provenance_sidecar(self):
        self.copernicus_path.with_suffix(".provenance.json").unlink()

        with self.assertRaisesRegex(
            MarineInputUnavailable,
            "Missing provenance sidecar",
        ):
            prepare_marine_prediction_inputs(self.start_time_iso)

    def test_rejects_incomplete_provenance(self):
        provenance_path = self.copernicus_path.with_suffix(".provenance.json")
        provenance_path.write_text(
            json.dumps({"provider": "Copernicus Marine"}),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            MarineInputUnavailable,
            "Incomplete provenance",
        ):
            prepare_marine_prediction_inputs(self.start_time_iso)

    def test_rejects_unavailable_exact_forecast_timestamp(self):
        unavailable_time = (self.start_time + timedelta(hours=2)).isoformat()

        with self.assertRaisesRegex(
            MarineInputUnavailable,
            "No Copernicus observation exists",
        ):
            prepare_marine_prediction_inputs(unavailable_time)

    def test_rejects_copernicus_grid_without_required_coverage(self):
        latitude = np.linspace(-4.8, -3.3, 5)
        longitude = np.linspace(40.2, 40.9, 5)

        self._write_copernicus_dataset(
            coverage=(latitude, longitude),
        )

        with self.assertRaisesRegex(
            MarineInputUnavailable,
            "Copernicus does not cover",
        ):
            prepare_marine_prediction_inputs(self.start_time_iso)

    def test_rejects_gebco_grid_without_required_coverage(self):
        latitude = np.linspace(-4.8, -3.3, 5)
        longitude = np.linspace(40.2, 40.9, 5)

        self._write_gebco_dataset(
            coverage=(latitude, longitude),
        )

        with self.assertRaisesRegex(
            MarineInputUnavailable,
            "GEBCO does not cover",
        ):
            prepare_marine_prediction_inputs(self.start_time_iso)

    def test_rejects_non_positive_offshore_bathymetry(self):
        self._write_gebco_dataset(elevation=0.0)

        with self.assertRaisesRegex(
            MarineInputUnavailable,
            "no valid underwater bathymetry",
        ):
            prepare_marine_prediction_inputs(self.start_time_iso)

    def test_rejects_unsupported_domain(self):
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported marine prediction domain",
        ):
            prepare_marine_prediction_inputs(
                self.start_time_iso,
                domain="mombasa_harbour",
            )
