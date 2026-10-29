from unittest.mock import patch

from django.test import TestCase

from port_twin_app.models import Scenario, SimulationResult, SimulationRun
from port_twin_app.services.simulation_execution import (
    execute_simulation_run,
)


class SimulationExecutionTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.scenario = Scenario.objects.create(
            name="Execution Test Scenario",
            description="Scenario used to test simulation persistence.",
            parameters={
                "simulation_hours": 24,
                "expected_vessel_arrivals": 1,
                "average_service_time_hours": 8,
                "arrival_interval_hours": 2,
                "available_berths": 2,
                "marine_conditions_enabled": True,
                "ai_predictions_enabled": False,
            },
            status="ready",
        )

    def setUp(self):
        self.simulation_run = SimulationRun.objects.create(
            scenario=self.scenario,
        )

    def test_successful_run_persists_results(self):
        with patch(
            "port_twin_app.services.simulation_execution.run_scenario_simulation"
        ) as mock_simulation:
            from port_twin_app.services.simulation import (
                SimulationOutput,
                SimulationSummary,
                VesselSimulationResult,
            )

            mock_simulation.return_value = SimulationOutput(
                vessel_results=(
                    VesselSimulationResult(
                        vessel_id="test-vessel-1",
                        arrival_hour=0,
                        berth_id="test-berth-1",
                        service_start_hour=0,
                        departure_hour=8,
                        waiting_hours=0,
                        status="completed",
                    ),
                ),
                summary=SimulationSummary(
                    total_arrivals=1,
                    vessels_served=1,
                    vessels_waiting=0,
                    vessels_in_service=0,
                    average_waiting_hours=0,
                    average_turnaround_hours=8,
                    berth_utilization_percent=33.33,
                ),
            )

            execute_simulation_run(self.simulation_run)

        self.simulation_run.refresh_from_db()

        self.assertEqual(self.simulation_run.status, "completed")
        self.assertIsNotNone(self.simulation_run.started_at)
        self.assertIsNotNone(self.simulation_run.completed_at)
        self.assertEqual(
            SimulationResult.objects.filter(simulation_run=self.simulation_run).count(),
            2,
        )

        self.assertEqual(
            SimulationResult.objects.get(
                simulation_run=self.simulation_run,
                result_type="summary",
            ).data["vessels_served"],
            1,
        )

    def test_failed_run_records_error(self):
        with (
            patch(
                "port_twin_app.services.simulation_execution.run_scenario_simulation",
                side_effect=ValueError("Test simulation failure"),
            ),
            self.assertRaisesMessage(
                ValueError,
                "Test simulation failure",
            ),
        ):
            execute_simulation_run(self.simulation_run)

        self.simulation_run.refresh_from_db()

        self.assertEqual(self.simulation_run.status, "failed")
        self.assertEqual(
            self.simulation_run.error_message,
            "Simulation execution failed. Check server logs for details.",
        )
        self.assertIsNotNone(self.simulation_run.completed_at)

        self.assertFalse(
            SimulationResult.objects.filter(simulation_run=self.simulation_run).exists()
        )

    def test_completed_run_cannot_be_executed_again(self):
        self.simulation_run.status = SimulationRun.Status.COMPLETED
        self.simulation_run.save(update_fields=["status"])

        with (
            patch(
                "port_twin_app.services.simulation_execution.run_scenario_simulation"
            ) as mock_simulation,
            self.assertRaisesMessage(
                ValueError,
                "Only pending simulation runs can be executed.",
            ),
        ):
            execute_simulation_run(self.simulation_run)

        mock_simulation.assert_not_called()

    def test_failed_run_cannot_be_executed_again(self):
        self.simulation_run.status = SimulationRun.Status.FAILED
        self.simulation_run.save(update_fields=["status"])

        with (
            patch(
                "port_twin_app.services.simulation_execution.run_scenario_simulation"
            ) as mock_simulation,
            self.assertRaisesMessage(
                ValueError,
                "Only pending simulation runs can be executed.",
            ),
        ):
            execute_simulation_run(self.simulation_run)

        mock_simulation.assert_not_called()

    def test_failed_run_does_not_persist_partial_results(self):
        from port_twin_app.services.simulation import (
            SimulationOutput,
            SimulationSummary,
        )

        output = SimulationOutput(
            vessel_results=(),
            summary=SimulationSummary(
                total_arrivals=0,
                vessels_served=0,
                vessels_waiting=0,
                vessels_in_service=0,
                average_waiting_hours=0,
                average_turnaround_hours=0,
                berth_utilization_percent=0,
            ),
        )

        with (
            patch(
                "port_twin_app.services.simulation_execution.run_scenario_simulation",
                return_value=output,
            ),
            patch(
                "port_twin_app.services.simulation_execution.SimulationResult.objects.create",
                side_effect=[None, RuntimeError("Result persistence failed")],
            ),
            self.assertRaisesMessage(
                RuntimeError,
                "Result persistence failed",
            ),
        ):
            execute_simulation_run(self.simulation_run)

        self.simulation_run.refresh_from_db()

        self.assertEqual(self.simulation_run.status, SimulationRun.Status.FAILED)
        self.assertEqual(
            self.simulation_run.error_message,
            "Simulation execution failed. Check server logs for details.",
        )
        self.assertFalse(
            SimulationResult.objects.filter(simulation_run=self.simulation_run).exists()
        )
