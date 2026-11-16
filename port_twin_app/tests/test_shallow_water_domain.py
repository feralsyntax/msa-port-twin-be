import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from django.test import SimpleTestCase

from port_twin_app.services.marine_data.shallow_water import (
    ShallowWaterConfig,
    load_shallow_water_domain,
)


class ShallowWaterDomainTests(SimpleTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        self.path = Path(self.temp_dir.name) / "bathymetry.nc"

        depth = np.full((5, 6), 20.0)
        depth[0, 0] = np.nan
        depth[1, 1] = np.nan

        # One isolated water cell should be excluded.
        depth[4, 5] = 10.0

        xr.Dataset(
            data_vars={
                "bathymetry_m": (
                    ("northing", "easting"),
                    depth,
                ),
                "source_coverage": (
                    ("northing", "easting"),
                    np.isfinite(depth),
                ),
            },
            coords={
                "easting": np.arange(6) * 250.0,
                "northing": np.arange(5) * 250.0,
            },
            attrs={
                "projected_crs": "EPSG:32737",
                "grid_resolution_m": 250,
                "source_vertical_reference": "height_above_mean_sea_level",
                "dataset_version": "GEBCO_2026",
            },
        ).to_netcdf(self.path, engine="scipy")

    def test_loads_largest_connected_water_component(self):
        domain = load_shallow_water_domain(self.path)

        self.assertEqual(domain.grid_spacing_m, 250)
        self.assertEqual(domain.projected_crs, "EPSG:32737")
        self.assertEqual(domain.total_water_cells, 28)
        self.assertEqual(domain.excluded_water_cells, 1)
        self.assertEqual(domain.water_mask.sum(), 27)
        self.assertTrue(np.isnan(domain.bathymetry_m[4, 5]))

    def test_records_source_checksum_and_vertical_reference(self):
        domain = load_shallow_water_domain(self.path)

        self.assertEqual(len(domain.source_sha256), 64)
        self.assertEqual(
            domain.source_vertical_reference,
            "height_above_mean_sea_level",
        )
        self.assertEqual(domain.source_dataset_version, "GEBCO_2026")

    def test_rejects_unsupported_crs(self):
        with xr.open_dataset(self.path) as dataset:
            source = dataset.load()

        source.attrs["projected_crs"] = "EPSG:4326"
        source.to_netcdf(self.path, engine="scipy")

        with self.assertRaisesMessage(ValueError, "Expected EPSG:32737"):
            load_shallow_water_domain(self.path)


class ShallowWaterConfigTests(SimpleTestCase):
    def test_default_configuration_is_reproducible(self):
        config = ShallowWaterConfig()

        self.assertEqual(config.random_seed, 42)
        self.assertEqual(config.tidal_boundary, "east")
        self.assertEqual(config.duration_seconds, 24 * 3600)

    def test_rejects_invalid_cfl_number(self):
        with self.assertRaisesMessage(ValueError, "CFL number"):
            ShallowWaterConfig(cfl_number=1.5)
