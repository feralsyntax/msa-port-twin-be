import json
from tempfile import TemporaryDirectory

import numpy as np
import xarray as xr
from django.test import SimpleTestCase

from port_twin_app.services.marine_data.preparation.mombasa_bathymetry import (
    build_mombasa_bathymetry,
    prepare_mombasa_bathymetry,
)


class SyntheticBathymetrySource:
    """Small in-memory GEBCO-like source for preparation tests."""

    def __init__(self):
        self.request = None

    def fetch(self, request):
        self.request = request

        longitude = np.array(
            [
                request.west,
                (request.west + request.east) / 2,
                request.east,
            ]
        )
        latitude = np.array(
            [
                request.south,
                (request.south + request.north) / 2,
                request.north,
            ]
        )

        elevation = np.full((3, 3), -20.0)
        tid = np.full((3, 3), 40, dtype=np.int16)

        return xr.Dataset(
            data_vars={
                "elevation_m": (
                    ("latitude", "longitude"),
                    elevation,
                    {"units": "m"},
                ),
                "tid": (
                    ("latitude", "longitude"),
                    tid,
                    {
                        "flag_meanings": (
                            "Land Singlebeam Multibeam predicted_altimetry"
                        ),
                        "flag_values": np.array(
                            [0, 10, 11, 40],
                            dtype=np.int16,
                        ),
                    },
                ),
            },
            coords={
                "longitude": longitude,
                "latitude": latitude,
            },
            attrs={
                "provider": "GEBCO",
                "source_dataset_id": "GEBCO",
                "source_dataset_version": "GEBCO_2026",
                "source": "GEBCO_2026",
                "vertical_reference": "Mean Sea Level",
                "source_file_path": "synthetic_gebco.nc",
                "tid_source_file_path": "synthetic_tid.nc",
            },
        )


class MombasaBathymetryPreparationTests(SimpleTestCase):
    def setUp(self):
        self.source = SyntheticBathymetrySource()

    def test_build_uses_configured_zone_bounds(self):
        dataset = build_mombasa_bathymetry(
            "port_operations",
            source=self.source,
        )

        self.assertEqual(
            self.source.request.west,
            39.5947,
        )
        self.assertEqual(
            self.source.request.east,
            39.7187,
        )
        self.assertEqual(dataset.attrs["domain_name"], "port_operations")

    def test_build_normalizes_bathymetry_to_positive_down(self):
        dataset = build_mombasa_bathymetry(
            "approaches",
            source=self.source,
        )

        self.assertTrue(np.all(dataset["bathymetry_m"].values == 20.0))
        self.assertEqual(
            dataset["bathymetry_m"].attrs["positive"],
            "down",
        )

    def test_build_preserves_elevation_and_tid(self):
        dataset = build_mombasa_bathymetry(
            "approaches",
            source=self.source,
        )

        self.assertIn("elevation_m", dataset)
        self.assertIn("tid", dataset)
        self.assertTrue(np.issubdtype(dataset["tid"].dtype, np.integer))

    def test_build_records_source_and_target_resolutions_separately(self):
        dataset = build_mombasa_bathymetry(
            "port_operations",
            source=self.source,
        )

        self.assertEqual(dataset.attrs["target_resolution_m"], 100)
        self.assertGreater(
            dataset.attrs["source_resolution_longitude_arc_seconds"],
            0,
        )

    def test_build_does_not_apply_vertical_datum_correction(self):
        dataset = build_mombasa_bathymetry(
            "approaches",
            source=self.source,
        )

        self.assertEqual(
            dataset.attrs["source_vertical_reference"],
            "Mean Sea Level",
        )
        self.assertEqual(
            dataset.attrs["vertical_datum_correction_applied"],
            "false",
        )
        self.assertEqual(
            dataset.attrs["vertical_datum_correction_m"],
            0.0,
        )

    def test_build_rejects_unknown_zone(self):
        with self.assertRaises(ValueError):
            build_mombasa_bathymetry(
                "unknown_zone",
                source=self.source,
            )

    def test_prepare_saves_netcdf_and_provenance(self):
        with TemporaryDirectory() as directory:
            result = prepare_mombasa_bathymetry(
                "port_operations",
                output_dir=directory,
                source=self.source,
            )

            self.assertTrue(result["output_path"].is_file())
            self.assertTrue(result["provenance_path"].is_file())

            provenance = json.loads(
                result["provenance_path"].read_text(encoding="utf-8")
            )

            self.assertEqual(
                provenance["domain_name"],
                "port_operations",
            )
            self.assertEqual(
                provenance["vertical_reference"],
                "Mean Sea Level",
            )
            self.assertFalse(provenance["vertical_datum_correction_applied"])
            self.assertEqual(
                provenance["target_resolution_m"],
                100,
            )

            with xr.open_dataset(
                result["output_path"],
                engine="scipy",
            ) as saved:
                self.assertIn("bathymetry_m", saved)
                self.assertIn("tid", saved)
