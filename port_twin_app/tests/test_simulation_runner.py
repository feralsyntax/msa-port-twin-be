from unittest.mock import patch

from django.test import TestCase

from port_twin_app.models import Berth, Scenario, Vessel
from port_twin_app.services.simulation_runner import run_scenario_simulation


class ScenarioSimulationRunnerTests(TestCase):
    def setUp(self):
        self.scenario = Scenario.objects.create(
            name="Test Scenario",
            parameters={
                "simulation_hours": 24,
                "expected_vessel_arrivals": 2,
                "arrival_interval_hours": 2,
                "average_service_time_hours": 8,
            },
        )

        self.vessel = Vessel.objects.create(
            name="Test Vessel",
            length_overall=200,
            draft=10,
        )

        self.berth = Berth.objects.create(
            name="Test Berth",
            max_vessel_length=250,
            max_vessel_draft=15,
            status=Berth.OperationalStatus.ACTIVE,
        )

    @patch("port_twin_app.services.simulation_runner.run_simulation")
    def test_builds_simulation_inputs_from_database_records(self, mock_run):
        run_scenario_simulation(self.scenario)

        vessels, berths, config = mock_run.call_args.args

        self.assertEqual(len(vessels), 2)
        self.assertEqual(vessels[0].length, 200)
        self.assertEqual(vessels[0].draft, 10)
        self.assertEqual(vessels[1].length, 200)
        self.assertEqual(vessels[1].draft, 10)
        self.assertNotEqual(vessels[0].vessel_id, vessels[1].vessel_id)

        self.assertEqual(len(berths), 1)
        self.assertEqual(berths[0].berth_id, str(self.berth.id))
        self.assertEqual(berths[0].max_vessel_length, 250)
        self.assertEqual(berths[0].max_vessel_draft, 15)

        self.assertEqual(config.duration_hours, 24)
        self.assertEqual(config.expected_arrivals, 2)
        self.assertEqual(config.arrival_interval_hours, 2)
        self.assertEqual(config.service_time_hours, 8)

    def test_rejects_negative_expected_arrivals(self):
        self.scenario.parameters["expected_vessel_arrivals"] = -1
        self.scenario.save()

        with self.assertRaisesMessage(
            ValueError,
            "Expected arrivals cannot be negative.",
        ):
            run_scenario_simulation(self.scenario)

    def test_requires_vessel_profile_when_arrivals_are_expected(self):
        self.vessel.delete()

        with self.assertRaisesMessage(
            ValueError,
            "At least one vessel profile is required to generate arrivals.",
        ):
            run_scenario_simulation(self.scenario)

    def test_rejects_negative_available_berths(self):
        self.scenario.parameters["available_berths"] = -1
        self.scenario.save()

        with self.assertRaisesMessage(
            ValueError,
            "Available berths cannot be negative.",
        ):
            run_scenario_simulation(self.scenario)

    @patch("port_twin_app.services.simulation_runner.run_simulation")
    def test_uses_only_active_berths(self, mock_run):
        Berth.objects.create(
            name="Inactive Berth",
            max_vessel_length=300,
            max_vessel_draft=20,
            status=Berth.OperationalStatus.INACTIVE,
        )

        run_scenario_simulation(self.scenario)

        _, berths, _ = mock_run.call_args.args

        self.assertEqual(len(berths), 1)
        self.assertEqual(berths[0].berth_id, str(self.berth.id))

    @patch("port_twin_app.services.simulation_runner.run_simulation")
    def test_returns_simulation_result(self, mock_run):
        expected_result = object()
        mock_run.return_value = expected_result

        result = run_scenario_simulation(self.scenario)

        self.assertIs(result, expected_result)
