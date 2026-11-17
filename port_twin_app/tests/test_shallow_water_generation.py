import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import xarray as xr
from django.test import SimpleTestCase

from port_twin_app.services.marine_data.generation import (
    generate_shallow_water_datasets,
    get_shallow_water_scenarios,
)


class ShallowWaterScenarioTests(SimpleTestCase):
    def test_returns_all_six_scenarios(self):
        scenarios = get_shallow_water_scenarios()

        self.assertEqual(len(scenarios), 6)

    def test_scenario_names_are_unique(self):
        scenarios = get_shallow_water_scenarios()
        names = [scenario.name for scenario in scenarios]

        self.assertEqual(len(names), len(set(names)))

    def test_scenarios_have_reproducible_configuration(self):
        scenarios = get_shallow_water_scenarios()

        for scenario in scenarios:
            with self.subTest(scenario=scenario.name):
                self.assertEqual(scenario.config.random_seed, 42)
                self.assertEqual(
                    scenario.config.duration_seconds,
                    6 * 3600,
                )

    def test_scenarios_can_be_selected(self):
        selected = get_shallow_water_scenarios(("stronger_tide", "calm_tidal_baseline"))

        self.assertEqual(
            [scenario.name for scenario in selected],
            ["stronger_tide", "calm_tidal_baseline"],
        )

    def test_unknown_scenario_is_rejected(self):
        with self.assertRaises(ValueError):
            get_shallow_water_scenarios(("unknown_scenario",))

    def test_scenarios_have_expected_forcing(self):
        scenarios = {
            scenario.name: scenario for scenario in get_shallow_water_scenarios()
        }

        self.assertEqual(
            scenarios["light_eastward_wind"].config.wind_u_ms,
            3.0,
        )
        self.assertEqual(
            scenarios["light_westward_wind"].config.wind_u_ms,
            -3.0,
        )
        self.assertEqual(
            scenarios["light_northward_wind"].config.wind_v_ms,
            3.0,
        )
        self.assertEqual(
            scenarios["stronger_tide"].config.tide_amplitude_m,
            0.8,
        )


class ShallowWaterDatasetGenerationTests(SimpleTestCase):
    def setUp(self):
        self.domain = SimpleNamespace(
            source_file="bathymetry.nc",
            source_sha256="test-checksum",
            source_dataset_version="GEBCO_2026",
            source_vertical_reference="height_above_mean_sea_level",
            projected_crs="EPSG:32737",
            grid_spacing_m=250.0,
            total_water_cells=100,
            excluded_water_cells=2,
        )

        self.dataset = xr.Dataset(
            data_vars={
                "water_level_m": (
                    ("time", "northing", "easting"),
                    [[[0.0]]],
                ),
                "current_u_ms": (
                    ("time", "northing", "easting"),
                    [[[0.1]]],
                ),
            },
            coords={
                "time": [0],
                "northing": [9500000.0],
                "easting": [580000.0],
            },
            attrs={"numerical_scheme": "SSP-RK2"},
        )

    @patch(
        "port_twin_app.services.marine_data.generation.dataset."
        "run_shallow_water_simulation"
    )
    @patch(
        "port_twin_app.services.marine_data.generation.dataset."
        "load_shallow_water_domain"
    )
    def test_generates_netcdf_and_provenance(
        self,
        mock_load_domain,
        mock_solver,
    ):
        mock_load_domain.return_value = self.domain
        mock_solver.return_value = self.dataset

        with tempfile.TemporaryDirectory() as temporary_directory:
            generated = generate_shallow_water_datasets(
                scenario_names=("calm_tidal_baseline",),
                output_directory=temporary_directory,
                duration_seconds=600,
            )

            netcdf_path = Path(generated["calm_tidal_baseline"])
            provenance_path = netcdf_path.with_name(
                "calm_tidal_baseline.provenance.json"
            )

            self.assertTrue(netcdf_path.exists())
            self.assertTrue(provenance_path.exists())

            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

            self.assertEqual(
                provenance["data_classification"],
                "simulated",
            )
            self.assertFalse(provenance["is_observed"])
            self.assertFalse(provenance["is_ground_truth"])
            self.assertEqual(
                provenance["bathymetry"]["source_sha256"],
                "test-checksum",
            )
            self.assertEqual(
                provenance["configuration"]["duration_seconds"],
                600,
            )

            with xr.open_dataset(
                netcdf_path,
                engine="h5netcdf",
            ) as exported:
                self.assertEqual(
                    exported.attrs["data_classification"],
                    "simulated",
                )
                self.assertEqual(
                    exported.attrs["scenario_name"],
                    "calm_tidal_baseline",
                )

        mock_load_domain.assert_called_once()
        mock_solver.assert_called_once()

    @patch(
        "port_twin_app.services.marine_data.generation.dataset."
        "run_shallow_water_simulation"
    )
    def test_duration_override_does_not_change_scenario_definition(
        self,
        mock_solver,
    ):
        mock_solver.return_value = self.dataset

        with tempfile.TemporaryDirectory() as temporary_directory:
            generate_shallow_water_datasets(
                scenario_names=("calm_tidal_baseline",),
                output_directory=temporary_directory,
                duration_seconds=600,
                domain=self.domain,
            )

        passed_config = mock_solver.call_args.args[1]
        original_scenario = get_shallow_water_scenarios(("calm_tidal_baseline",))[0]

        self.assertEqual(passed_config.duration_seconds, 600)
        self.assertEqual(
            original_scenario.config.duration_seconds,
            6 * 3600,
        )

    def test_rejects_non_positive_duration(self):
        with tempfile.TemporaryDirectory() as temporary_directory:  # noqa: SIM117
            with self.assertRaises(ValueError):
                generate_shallow_water_datasets(
                    scenario_names=("calm_tidal_baseline",),
                    output_directory=temporary_directory,
                    duration_seconds=0,
                    domain=self.domain,
                )

    def test_rejects_empty_scenario_selection(self):
        with tempfile.TemporaryDirectory() as temporary_directory:  # noqa: SIM117
            with self.assertRaises(ValueError):
                generate_shallow_water_datasets(
                    scenario_names=(),
                    output_directory=temporary_directory,
                    domain=self.domain,
                )
