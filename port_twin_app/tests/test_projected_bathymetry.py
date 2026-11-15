import numpy as np
import xarray as xr
from django.test import SimpleTestCase

from port_twin_app.services.marine_data.domains.mombasa_channel import (
    ZONES,
)
from port_twin_app.services.marine_data.preparation.projected_bathymetry import (
    build_projected_bathymetry_grid,
)


def synthetic_source(zone):
    """Create a small GEBCO-like source grid for a configured zone."""
    longitude = np.linspace(zone.bounds.west, zone.bounds.east, 5)
    latitude = np.linspace(zone.bounds.south, zone.bounds.north, 5)

    elevation = np.full((5, 5), -30.0, dtype=np.float32)
    elevation[0, 0] = 5.0
    elevation[0, 1] = 10.0

    tid = np.full((5, 5), 40, dtype=np.int16)
    tid[0, 0] = 0
    tid[0, 1] = 10

    return xr.Dataset(
        data_vars={
            "elevation_m": (
                ("latitude", "longitude"),
                elevation,
                {"units": "m"},
            ),
            "bathymetry_m": (
                ("latitude", "longitude"),
                np.where(elevation < 0, -elevation, np.nan),
                {"units": "m", "positive": "down"},
            ),
            "tid": (
                ("latitude", "longitude"),
                tid,
                {
                    "flag_values": np.array([0, 10, 40], dtype=np.int16),
                    "flag_meanings": "land singlebeam predicted",
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
            "vertical_reference": "Mean Sea Level",
        },
    )


class ProjectedBathymetryTests(SimpleTestCase):
    def test_builds_projected_grid_in_configured_crs(self):
        zone = next(zone for zone in ZONES if zone.name == "port_operations")

        grid = build_projected_bathymetry_grid(
            zone.name,
            source_dataset=synthetic_source(zone),
        )

        self.assertEqual(grid.attrs["projected_crs"], "EPSG:32737")
        self.assertEqual(grid.attrs["grid_resolution_m"], 100)
        self.assertIn("easting", grid.coords)
        self.assertIn("northing", grid.coords)
        self.assertIn("spatial_ref", grid)

    def test_grid_coordinates_follow_configured_resolution(self):
        zone = next(zone for zone in ZONES if zone.name == "approaches")

        grid = build_projected_bathymetry_grid(
            zone.name,
            source_dataset=synthetic_source(zone),
        )

        resolution = zone.target_resolution_m

        for coordinate in ("easting", "northing"):
            values = grid[coordinate].values

            self.assertTrue(np.allclose(np.diff(values), resolution))
            self.assertTrue(
                np.allclose(values / resolution, np.round(values / resolution))
            )

    def test_tid_uses_original_categories_without_interpolation(self):
        zone = next(zone for zone in ZONES if zone.name == "port_operations")

        grid = build_projected_bathymetry_grid(
            zone.name,
            source_dataset=synthetic_source(zone),
        )

        valid_tid = grid.tid.values[np.isfinite(grid.tid.values)]

        self.assertTrue(set(np.unique(valid_tid)).issubset({0.0, 10.0, 40.0}))
        self.assertEqual(grid.tid.attrs["resampling_method"], "nearest")

    def test_bathymetry_is_positive_down_and_land_is_masked(self):
        zone = next(zone for zone in ZONES if zone.name == "approaches")

        grid = build_projected_bathymetry_grid(
            zone.name,
            source_dataset=synthetic_source(zone),
        )

        valid_depths = grid.bathymetry_m.values[np.isfinite(grid.bathymetry_m.values)]

        self.assertTrue(valid_depths.size > 0)
        self.assertTrue(np.all(valid_depths > 0))
        self.assertTrue(np.isnan(grid.bathymetry_m.values).any())

    def test_source_coverage_masks_extrapolated_values(self):
        zone = next(zone for zone in ZONES if zone.name == "port_operations")
        source = synthetic_source(zone)

        # Remove source points around one corner to create a predictable
        # region outside the convex hull of valid elevation observations.
        source["elevation_m"].values[:2, :2] = np.nan
        source["bathymetry_m"].values[:2, :2] = np.nan

        grid = build_projected_bathymetry_grid(
            zone.name,
            source_dataset=source,
        )

        outside = ~grid.source_coverage.values

        self.assertTrue(outside.any())
        self.assertTrue(np.isnan(grid.elevation_m.values[outside]).all())
        self.assertTrue(np.isnan(grid.tid.values[outside]).all())

    def test_nested_grids_share_aligned_coordinates(self):
        zones = {zone.name: zone for zone in ZONES}

        parent = build_projected_bathymetry_grid(
            "approaches",
            source_dataset=synthetic_source(zones["approaches"]),
        )
        child = build_projected_bathymetry_grid(
            "port_operations",
            source_dataset=synthetic_source(zones["port_operations"]),
        )

        shared_easting = np.intersect1d(
            parent.easting.values,
            child.easting.values,
        )
        shared_northing = np.intersect1d(
            parent.northing.values,
            child.northing.values,
        )

        self.assertGreater(shared_easting.size, 0)
        self.assertGreater(shared_northing.size, 0)

        # The least common multiple of 250 m and 100 m is 500 m.
        self.assertTrue(np.allclose(shared_easting % 500, 0))
        self.assertTrue(np.allclose(shared_northing % 500, 0))

    def test_preserves_vertical_reference_without_correction(self):
        zone = next(zone for zone in ZONES if zone.name == "port_operations")

        grid = build_projected_bathymetry_grid(
            zone.name,
            source_dataset=synthetic_source(zone),
        )

        self.assertEqual(
            grid.attrs["source_vertical_reference"],
            "Mean Sea Level",
        )
        self.assertFalse(grid.attrs["vertical_datum_correction_applied"])
        self.assertEqual(grid.attrs["vertical_datum_correction_m"], 0.0)
        self.assertFalse(grid.attrs["source_accuracy_claim"])

    def test_land_source_cells_are_excluded_from_marine_coverage(self):
        zone = next(zone for zone in ZONES if zone.name == "port_operations")

        grid = build_projected_bathymetry_grid(
            zone.name,
            source_dataset=synthetic_source(zone),
        )

        outside = ~grid.source_coverage.values

        self.assertTrue(outside.any())
        self.assertTrue(np.isnan(grid.bathymetry_m.values[outside]).all())
        self.assertTrue(np.isnan(grid.elevation_m.values[outside]).all())
        self.assertTrue(np.isnan(grid.tid.values[outside]).all())
