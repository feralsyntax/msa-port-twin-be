from types import SimpleNamespace

import numpy as np
from django.test import SimpleTestCase

from port_twin_app.services.marine_data.shallow_water import (
    ShallowWaterConfig,
)
from port_twin_app.services.marine_data.shallow_water.solver import (
    _face_flux,
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
                "water_volume_m3",
                "boundary_volume_exchange_m3",
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

    def test_wet_dry_cells_remain_properly_masked(self):
        domain = synthetic_domain()
        domain.water_mask[:, 4:6] = False
        domain.bathymetry_m[:, 4:6] = np.nan

        config = ShallowWaterConfig(
            duration_seconds=600,
            output_interval_seconds=120,
            tide_amplitude_m=0.2,
            tide_period_seconds=1200,
            wind_u_ms=0.0,
            wind_v_ms=0.0,
        )

        result = run_shallow_water_simulation(domain, config)

        land_mask = ~domain.water_mask

        self.assertTrue(np.isnan(result.water_level_m.values[:, land_mask]).all())
        self.assertTrue(np.isnan(result.current_u_ms.values[:, land_mask]).all())
        self.assertTrue(np.isnan(result.current_v_ms.values[:, land_mask]).all())

        self.assertTrue(
            np.isfinite(result.current_u_ms.values[:, domain.water_mask]).all()
        )
        self.assertTrue(
            np.isfinite(result.current_v_ms.values[:, domain.water_mask]).all()
        )

    def test_internal_face_uses_identical_mass_flux_on_both_sides(self):
        eta_l = np.array([[0.1, 0.2]])
        hu_l = np.array([[1.0, 2.0]])
        hv_l = np.zeros((1, 2))
        bed_l = np.array([[10.0, 12.0]])
        wet_l = np.ones((1, 2), dtype=bool)

        eta_r = np.array([[0.0, 0.1]])
        hu_r = np.array([[0.5, 1.5]])
        hv_r = np.zeros((1, 2))
        bed_r = np.array([[20.0, 8.0]])
        wet_r = np.ones((1, 2), dtype=bool)

        left_flux, right_flux = _face_flux(
            eta_l,
            hu_l,
            hv_l,
            bed_l,
            wet_l,
            eta_r,
            hu_r,
            hv_r,
            bed_r,
            wet_r,
            gravity=9.81,
            normal="x",
        )

        self.assertTrue(np.allclose(left_flux[0], right_flux[0]))
        self.assertGreater(
            np.max(np.abs(left_flux[0])),
            1e-6,
        )

    def test_longer_tidal_simulation_remains_stable(self):
        config = ShallowWaterConfig(
            duration_seconds=1800,
            output_interval_seconds=300,
            tide_amplitude_m=0.2,
            tide_period_seconds=1200,
            wind_u_ms=0.0,
            wind_v_ms=0.0,
        )

        result = run_shallow_water_simulation(
            synthetic_domain(),
            config,
        )

        currents_u = result.current_u_ms.values
        currents_v = result.current_v_ms.values

        self.assertTrue(np.isfinite(currents_u).all())
        self.assertTrue(np.isfinite(currents_v).all())
        self.assertLess(np.max(np.abs(currents_u)), 5.0)
        self.assertLess(np.max(np.abs(currents_v)), 5.0)

    def test_tidal_boundary_responds_in_correct_direction(self):
        domain = synthetic_domain()

        common_config = {
            "duration_seconds": 120,
            "output_interval_seconds": 30,
            "tide_period_seconds": 1200,
            "tide_amplitude_m": 0.5,
            "wind_u_ms": 0.0,
            "wind_v_ms": 0.0,
            "bottom_drag_coefficient": 0.0,
            "coriolis_parameter_s1": 0.0,
        }

        rising = run_shallow_water_simulation(
            domain,
            ShallowWaterConfig(
                **common_config,
                tide_phase_radians=0.0,
            ),
        )

        falling = run_shallow_water_simulation(
            domain,
            ShallowWaterConfig(
                **common_config,
                tide_phase_radians=np.pi,
            ),
        )

        rising_level = rising["water_level_m"].isel(time_seconds=-1).values

        falling_level = falling["water_level_m"].isel(time_seconds=-1).values

        assert np.nanmax(rising_level) > 0
        assert np.nanmin(falling_level) < 0

    def test_water_volume_change_matches_boundary_exchange(self):
        config = ShallowWaterConfig(
            duration_seconds=120,
            output_interval_seconds=30,
            tide_period_seconds=1200,
            tide_amplitude_m=0.5,
            tide_phase_radians=0.0,
            wind_u_ms=0.0,
            wind_v_ms=0.0,
            bottom_drag_coefficient=0.0,
            coriolis_parameter_s1=0.0,
        )

        result = run_shallow_water_simulation(synthetic_domain(), config)

        volume_change = (
            result["water_volume_m3"].values - result["water_volume_m3"].values[0]
        )
        boundary_exchange = result["boundary_volume_exchange_m3"].values

        np.testing.assert_allclose(
            volume_change,
            boundary_exchange,
            rtol=1e-9,
            atol=1e-4,
        )
