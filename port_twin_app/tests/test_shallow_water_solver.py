from types import SimpleNamespace

import numpy as np
from django.test import SimpleTestCase

from port_twin_app.services.marine_data.shallow_water import (
    ShallowWaterConfig,
)
from port_twin_app.services.marine_data.shallow_water.solver import (
    run_shallow_water_simulation,
)


def synthetic_domain():
    """Create a small, flat-bottomed synthetic water domain."""

    ny, nx = 8, 10
    depth = np.full((ny, nx), 20.0, dtype=np.float64)
    mask = np.ones((ny, nx), dtype=bool)

    return SimpleNamespace(
        bathymetry_m=depth,
        water_mask=mask,
        easting_m=np.arange(nx, dtype=np.float64) * 250.0,
        northing_m=np.arange(ny, dtype=np.float64) * 250.0,
        grid_spacing_m=250.0,
        projected_crs="EPSG:32737",
        source_vertical_reference="height_above_mean_sea_level",
        source_dataset_version="synthetic-test",
        source_sha256="a" * 64,
    )


class ShallowWaterSolverTests(SimpleTestCase):
    def test_resting_water_remains_stationary(self):
        config = ShallowWaterConfig(
            duration_seconds=120,
            output_interval_seconds=60,
            tide_amplitude_m=0.0,
            wind_u_ms=0.0,
            wind_v_ms=0.0,
            bottom_drag_coefficient=0.0,
            coriolis_parameter_s1=0.0,
        )

        result = run_shallow_water_simulation(
            synthetic_domain(),
            config,
        )

        self.assertTrue(np.allclose(result.water_level_m.values, 0.0, atol=1e-6))
        self.assertTrue(np.allclose(result.current_u_ms.values, 0.0, atol=1e-6))
        self.assertTrue(np.allclose(result.current_v_ms.values, 0.0, atol=1e-6))

    def test_tidal_boundary_generates_currents(self):
        config = ShallowWaterConfig(
            duration_seconds=600,
            output_interval_seconds=60,
            tide_amplitude_m=0.2,
            tide_period_seconds=1200,
            tide_phase_radians=0.0,
            wind_u_ms=0.0,
            wind_v_ms=0.0,
            bottom_drag_coefficient=0.0,
            coriolis_parameter_s1=0.0,
        )

        result = run_shallow_water_simulation(
            synthetic_domain(),
            config,
        )

        currents = result.current_u_ms.values

        self.assertTrue(np.isfinite(currents).all())
        self.assertGreater(np.max(np.abs(currents)), 1e-7)

    def test_output_is_explicitly_labelled_simulated(self):
        result = run_shallow_water_simulation(
            synthetic_domain(),
            ShallowWaterConfig(
                duration_seconds=60,
                output_interval_seconds=60,
                tide_amplitude_m=0.0,
                wind_u_ms=0.0,
                wind_v_ms=0.0,
            ),
        )

        self.assertEqual(result.attrs["data_classification"], "simulated")
        self.assertFalse(result.attrs["is_observed"])
        self.assertFalse(result.attrs["is_ground_truth"])

    def test_output_contains_expected_variables_and_dimensions(self):
        result = run_shallow_water_simulation(
            synthetic_domain(),
            ShallowWaterConfig(
                duration_seconds=120,
                output_interval_seconds=60,
                tide_amplitude_m=0.0,
                wind_u_ms=0.0,
                wind_v_ms=0.0,
            ),
        )

        self.assertEqual(
            set(result.data_vars),
            {
                "water_level_m",
                "current_u_ms",
                "current_v_ms",
                "bathymetry_m",
                "water_mask",
            },
        )

        self.assertEqual(result.sizes["time_seconds"], 3)
        self.assertEqual(result.sizes["northing"], 8)
        self.assertEqual(result.sizes["easting"], 10)

    def test_rejects_invalid_bathymetry(self):
        domain = synthetic_domain()
        domain.bathymetry_m[0, 0] = np.nan

        with self.assertRaisesMessage(
            ValueError,
            "finite and positive down",
        ):
            run_shallow_water_simulation(
                domain,
                ShallowWaterConfig(
                    duration_seconds=60,
                    output_interval_seconds=60,
                ),
            )
