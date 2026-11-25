from types import SimpleNamespace
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

    def run_ais_scenario(self):
        self.scenario.parameters = {
            "simulation_hours": 24,
            "average_service_time_hours": 8,
            "arrival_source": "ais",
        }
        self.scenario.save()
        return run_scenario_simulation(self.scenario)

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

    def test_requires_valid_vessel_profile_when_arrivals_are_expected(self):
        self.vessel.delete()

        with self.assertRaisesMessage(
            ValueError,
            "At least one vessel profile with valid length and draft "
            "is required to generate arrivals.",
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

    def test_rejects_non_dictionary_scenario_parameters(self):
        self.scenario.parameters = []
        self.scenario.save()

        with self.assertRaisesMessage(
            ValueError,
            "Scenario parameters must be a dictionary.",
        ):
            run_scenario_simulation(self.scenario)

    def test_rejects_missing_required_scenario_parameter(self):
        del self.scenario.parameters["simulation_hours"]
        self.scenario.save()

        with self.assertRaisesMessage(
            ValueError,
            "Missing required scenario parameter: simulation_hours.",
        ):
            run_scenario_simulation(self.scenario)

    def test_rejects_invalid_expected_arrivals_type(self):
        self.scenario.parameters["expected_vessel_arrivals"] = 1.5
        self.scenario.save()

        with self.assertRaisesMessage(
            ValueError,
            "Expected arrivals must be a non-negative integer.",
        ):
            run_scenario_simulation(self.scenario)

    def test_rejects_boolean_expected_arrivals(self):
        self.scenario.parameters["expected_vessel_arrivals"] = True
        self.scenario.save()

        with self.assertRaisesMessage(
            ValueError,
            "Expected arrivals must be a non-negative integer.",
        ):
            run_scenario_simulation(self.scenario)

    def test_rejects_invalid_available_berths_type(self):
        self.scenario.parameters["available_berths"] = 1.5
        self.scenario.save()

        with self.assertRaisesMessage(
            ValueError,
            "Available berths must be a non-negative integer.",
        ):
            run_scenario_simulation(self.scenario)

    def test_rejects_boolean_available_berths(self):
        self.scenario.parameters["available_berths"] = True
        self.scenario.save()

        with self.assertRaisesMessage(
            ValueError,
            "Available berths must be a non-negative integer.",
        ):
            run_scenario_simulation(self.scenario)

    @patch("port_twin_app.services.simulation_runner.run_simulation")
    def test_skips_vessel_profiles_with_missing_dimensions(self, mock_run):
        self.vessel.length_overall = None
        self.vessel.save()

        valid_vessel = Vessel.objects.create(
            name="Valid Vessel",
            length_overall=180,
            draft=9,
        )

        run_scenario_simulation(self.scenario)

        vessels, _, _ = mock_run.call_args.args

        self.assertEqual(len(vessels), 2)
        self.assertTrue(
            all(vessel.length == 180 and vessel.draft == 9 for vessel in vessels)
        )
        self.assertTrue(
            all(
                vessel.vessel_id.startswith(f"{valid_vessel.id}-arrival-")
                for vessel in vessels
            )
        )

    def test_rejects_when_all_vessel_profiles_have_invalid_dimensions(self):
        self.vessel.length_overall = None
        self.vessel.save()

        Vessel.objects.create(
            name="Invalid Vessel",
            length_overall=0,
            draft=10,
        )

        with self.assertRaisesMessage(
            ValueError,
            "At least one vessel profile with valid length and draft "
            "is required to generate arrivals.",
        ):
            run_scenario_simulation(self.scenario)

    def test_rejects_berth_with_missing_draft_capacity(self):
        self.berth.max_vessel_draft = None
        self.berth.save()

        with self.assertRaisesMessage(
            ValueError,
            "Active berths must have valid length and draft capacity limits.",
        ):
            run_scenario_simulation(self.scenario)

    @patch("port_twin_app.services.simulation_runner.run_simulation")
    def test_allows_zero_available_berths(self, mock_run):
        self.scenario.parameters["available_berths"] = 0
        self.scenario.save()

        run_scenario_simulation(self.scenario)

        _, berths, _ = mock_run.call_args.args
        self.assertEqual(berths, [])

    @patch("port_twin_app.services.simulation_runner.run_simulation")
    @patch("port_twin_app.services.simulation_runner.estimate_vessel_eta")
    def test_uses_ais_eta_and_real_vessel_ids(self, mock_eta, mock_run):
        second_vessel = Vessel.objects.create(
            name="Second Vessel",
            length_overall=180,
            draft=9,
        )

        mock_eta.side_effect = lambda vessel, now: SimpleNamespace(
            eta_hours=(3.5 if vessel.pk == self.vessel.pk else 1.5)
        )

        self.run_ais_scenario()

        vessels, _, config = mock_run.call_args.args

        self.assertEqual(len(vessels), 2)

        # Arrivals should be ordered by ETA.
        self.assertEqual(vessels[0].vessel_id, str(second_vessel.pk))
        self.assertEqual(vessels[0].arrival_hour, 1.5)

        self.assertEqual(vessels[1].vessel_id, str(self.vessel.pk))
        self.assertEqual(vessels[1].arrival_hour, 3.5)

        self.assertEqual(config.expected_arrivals, 2)
        self.assertIsNone(config.arrival_interval_hours)

    @patch("port_twin_app.services.simulation_runner.run_simulation")
    @patch("port_twin_app.services.simulation_runner.estimate_vessel_eta")
    def test_excludes_vessels_without_reliable_eta(self, mock_eta, mock_run):
        second_vessel = Vessel.objects.create(
            name="Unreliable Vessel",
            length_overall=180,
            draft=9,
        )

        mock_eta.side_effect = lambda vessel, now: (
            SimpleNamespace(eta_hours=4) if vessel.pk == self.vessel.pk else None
        )

        self.run_ais_scenario()

        vessels, _, config = mock_run.call_args.args

        self.assertEqual(len(vessels), 1)
        self.assertEqual(vessels[0].vessel_id, str(self.vessel.pk))
        self.assertNotEqual(vessels[0].vessel_id, str(second_vessel.pk))
        self.assertEqual(config.expected_arrivals, 1)

    @patch("port_twin_app.services.simulation_runner.run_simulation")
    @patch("port_twin_app.services.simulation_runner.estimate_vessel_eta")
    def test_excludes_vessels_already_in_port_operations(
        self,
        mock_eta,
        mock_run,
    ):
        mock_eta.return_value = None

        self.run_ais_scenario()

        vessels, _, config = mock_run.call_args.args

        self.assertEqual(vessels, [])
        self.assertEqual(config.expected_arrivals, 0)

    @patch("port_twin_app.services.simulation_runner.run_simulation")
    @patch("port_twin_app.services.simulation_runner.estimate_vessel_eta")
    def test_excludes_etas_outside_simulation_window(
        self,
        mock_eta,
        mock_run,
    ):
        mock_eta.return_value = SimpleNamespace(eta_hours=24)

        self.run_ais_scenario()

        vessels, _, config = mock_run.call_args.args

        self.assertEqual(vessels, [])
        self.assertEqual(config.expected_arrivals, 0)

    @patch("port_twin_app.services.simulation_runner.run_simulation")
    @patch("port_twin_app.services.simulation_runner.estimate_vessel_eta")
    def test_handles_no_eligible_ais_vessels(self, mock_eta, mock_run):
        mock_eta.return_value = None

        self.run_ais_scenario()

        vessels, _, config = mock_run.call_args.args

        self.assertEqual(vessels, [])
        self.assertEqual(config.expected_arrivals, 0)
        self.assertIsNone(config.arrival_interval_hours)
        self.assertEqual(mock_eta.call_count, 1)

    def set_stochastic_scenario(self, **overrides):
        parameters = {
            "simulation_hours": 24,
            "expected_vessel_arrivals": 12,
            "average_service_time_hours": 8,
            "arrival_source": "stochastic",
            "random_seed": 42,
        }
        parameters.update(overrides)

        self.scenario.parameters = parameters
        self.scenario.save()

    @patch("port_twin_app.services.simulation_runner.run_simulation")
    def test_generates_stochastic_arrivals_within_simulation_horizon(self, mock_run):
        self.set_stochastic_scenario()

        run_scenario_simulation(self.scenario)

        vessels, _, config = mock_run.call_args.args

        self.assertGreater(len(vessels), 0)
        self.assertTrue(
            all(0 < vessel.arrival_hour < config.duration_hours for vessel in vessels)
        )
        self.assertEqual(config.expected_arrivals, len(vessels))
        self.assertIsNone(config.arrival_interval_hours)

    @patch("port_twin_app.services.simulation_runner.run_simulation")
    def test_stochastic_arrivals_are_reproducible(self, mock_run):
        self.set_stochastic_scenario()

        run_scenario_simulation(self.scenario)
        first_vessels = mock_run.call_args.args[0]

        run_scenario_simulation(self.scenario)
        second_vessels = mock_run.call_args.args[0]

        first_results = [
            (
                vessel.vessel_id,
                vessel.arrival_hour,
                vessel.service_time_hours,
            )
            for vessel in first_vessels
        ]
        second_results = [
            (
                vessel.vessel_id,
                vessel.arrival_hour,
                vessel.service_time_hours,
            )
            for vessel in second_vessels
        ]

        self.assertEqual(first_results, second_results)

    @patch("port_twin_app.services.simulation_runner.run_simulation")
    def test_assigns_uniform_service_times_within_configured_bounds(self, mock_run):
        self.set_stochastic_scenario(
            service_time_distribution="uniform",
            minimum_service_time_hours=4,
            maximum_service_time_hours=10,
        )

        run_scenario_simulation(self.scenario)

        vessels, _, _ = mock_run.call_args.args

        self.assertGreater(len(vessels), 0)

        for vessel in vessels:
            self.assertGreaterEqual(vessel.service_time_hours, 4)
            self.assertLessEqual(vessel.service_time_hours, 10)

    @patch("port_twin_app.services.simulation_runner.run_simulation")
    def test_passes_vessel_types_and_priorities_to_simulation(self, mock_run):
        self.vessel.vessel_type = Vessel.VesselType.TANKER
        self.vessel.save()

        priorities = {
            "tanker": 0,
            "container": 1,
            "bulk_carrier": 2,
        }

        self.set_stochastic_scenario(vessel_type_priorities=priorities)

        run_scenario_simulation(self.scenario)

        vessels, _, config = mock_run.call_args.args

        self.assertTrue(all(v.vessel_type == "tanker" for v in vessels))
        self.assertEqual(config.vessel_type_priorities, priorities)

    @patch("port_twin_app.services.simulation_runner.run_simulation")
    def test_allows_zero_stochastic_arrivals(self, mock_run):
        self.set_stochastic_scenario(
            expected_vessel_arrivals=1,
            random_seed=42,
        )

        run_scenario_simulation(self.scenario)

        vessels, _, config = mock_run.call_args.args

        self.assertEqual(vessels, [])
        self.assertEqual(config.expected_arrivals, 0)
