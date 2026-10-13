from unittest.mock import patch

from django.test import TestCase
from rest_framework.test import APIClient

from port_twin_app.models import Scenario, SimulationResult, SimulationRun
from port_twin_app.services.simulation import (
    SimulationOutput,
    SimulationSummary,
    VesselSimulationResult,
)


class SimulationAPITests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.scenario = Scenario.objects.create(
            name="API Test Scenario",
            description="Scenario used to test the simulation API.",
            parameters={
                "simulation_hours": 24,
                "expected_vessel_arrivals": 1,
                "average_service_time_hours": 8,
                "arrival_interval_hours": 2,
                "available_berths": 2,
            },
            status="ready",
        )

    def setUp(self):
        self.client = APIClient()
        self.url = f"/api/v1/scenarios/{self.scenario.id}/simulations/"

    @patch("port_twin_app.services.simulation_execution.run_scenario_simulation")
    def test_successful_simulation_request(self, mock_simulation):
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

        response = self.client.post(self.url)

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["status"], "completed")
        self.assertEqual(
            response.data["scenario_id"],
            str(self.scenario.id),
        )
        self.assertEqual(response.data["summary"]["vessels_served"], 1)

        simulation_run = SimulationRun.objects.get(id=response.data["id"])
        self.assertEqual(simulation_run.scenario, self.scenario)
        self.assertEqual(
            SimulationResult.objects.filter(simulation_run=simulation_run).count(),
            2,
        )

    def test_nonexistent_scenario_returns_404(self):
        import uuid

        response = self.client.post(f"/api/v1/scenarios/{uuid.uuid4()}/simulations/")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(SimulationRun.objects.count(), 0)

    @patch(
        "port_twin_app.services.simulation_execution.run_scenario_simulation",
        side_effect=ValueError("Invalid simulation configuration"),
    )
    def test_validation_failure_returns_400(self, mock_simulation):
        response = self.client.post(self.url)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["status"], "failed")
        self.assertEqual(
            response.data["error"],
            "Invalid simulation configuration",
        )

        simulation_run = SimulationRun.objects.get(scenario=self.scenario)
        self.assertEqual(simulation_run.status, "failed")
        self.assertFalse(
            SimulationResult.objects.filter(simulation_run=simulation_run).exists()
        )
