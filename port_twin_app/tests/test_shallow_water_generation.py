from django.test import SimpleTestCase

from port_twin_app.services.marine_data.generation import (
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
