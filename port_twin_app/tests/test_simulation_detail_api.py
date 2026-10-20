import uuid

from django.test import TestCase
from rest_framework.test import APIClient

from port_twin_app.models import Scenario, SimulationResult, SimulationRun


class SimulationDetailAPITests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.scenario = Scenario.objects.create(
            name="Detail API Test Scenario",
            description="Scenario used to test simulation retrieval.",
            parameters={},
            status="ready",
        )

        cls.simulation_run = SimulationRun.objects.create(
            scenario=cls.scenario,
            status="completed",
        )

        SimulationResult.objects.create(
            simulation_run=cls.simulation_run,
            result_type="summary",
            data={
                "total_arrivals": 1,
                "vessels_served": 1,
                "vessels_waiting": 0,
                "vessels_in_service": 0,
                "average_waiting_hours": 0.0,
                "average_turnaround_hours": 8.0,
                "berth_utilization_percent": 33.33,
            },
        )

        SimulationResult.objects.create(
            simulation_run=cls.simulation_run,
            result_type="vessel_results",
            data={
                "vessels": [
                    {
                        "vessel_id": "test-vessel-1",
                        "arrival_hour": 0.0,
                        "berth_id": "test-berth-1",
                        "service_start_hour": 0.0,
                        "departure_hour": 8.0,
                        "waiting_hours": 0.0,
                        "status": "completed",
                        "reason": "No compatible berth was available.",
                    }
                ]
            },
        )

    def setUp(self):
        self.client = APIClient()
        self.url = f"/api/v1/simulations/{self.simulation_run.id}/"

    def test_retrieve_simulation_with_results(self):
        response = self.client.get(self.url)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.data["id"],
            str(self.simulation_run.id),
        )
        self.assertEqual(
            response.data["scenario_name"],
            self.scenario.name,
        )
        self.assertEqual(
            response.data["summary"]["vessels_served"],
            1,
        )
        self.assertEqual(
            response.data["vessel_results"]["vessels"][0]["vessel_id"],
            "test-vessel-1",
        )
        self.assertEqual(
            response.data["vessel_results"]["vessels"][0]["reason"],
            "No compatible berth was available.",
        )

    def test_nonexistent_simulation_returns_404(self):
        response = self.client.get(f"/api/v1/simulations/{uuid.uuid4()}/")

        self.assertEqual(response.status_code, 404)
