import uuid
from unittest.mock import patch

from django.test import TestCase
from rest_framework.test import APIClient

from port_twin_app.models import Scenario, SimulationResult, SimulationRun


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

    @patch("port_twin_app.views.execute_simulation_task.delay")
    def test_successful_simulation_request_queues_run(self, mock_delay):
        response = self.client.post(self.url)

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.data["status"], "pending")
        self.assertEqual(
            response.data["scenario_id"],
            str(self.scenario.id),
        )
        self.assertIsNone(response.data["started_at"])
        self.assertIsNone(response.data["completed_at"])
        self.assertIsNone(response.data["summary"])

        simulation_run = SimulationRun.objects.get(id=response.data["id"])
        self.assertEqual(simulation_run.scenario, self.scenario)
        self.assertEqual(simulation_run.status, SimulationRun.Status.PENDING)
        self.assertFalse(
            SimulationResult.objects.filter(simulation_run=simulation_run).exists()
        )

        mock_delay.assert_called_once_with(str(simulation_run.pk))

    def test_nonexistent_scenario_returns_404(self):
        response = self.client.post(f"/api/v1/scenarios/{uuid.uuid4()}/simulations/")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(SimulationRun.objects.count(), 0)

    @patch(
        "port_twin_app.views.execute_simulation_task.delay",
        side_effect=ConnectionError("Redis unavailable"),
    )
    def test_queue_failure_marks_run_failed(self, mock_delay):
        response = self.client.post(self.url)

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.data["status"], "failed")
        self.assertEqual(
            response.data["error"],
            "Simulation could not be queued. Please try again.",
        )

        simulation_run = SimulationRun.objects.get(scenario=self.scenario)
        self.assertEqual(simulation_run.status, SimulationRun.Status.FAILED)
        self.assertIsNotNone(simulation_run.completed_at)
        self.assertEqual(
            simulation_run.error_message,
            "Simulation could not be queued. Please try again.",
        )
        self.assertFalse(
            SimulationResult.objects.filter(simulation_run=simulation_run).exists()
        )

        mock_delay.assert_called_once()
