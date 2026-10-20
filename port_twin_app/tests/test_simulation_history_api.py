from django.test import TestCase
from rest_framework.test import APIClient

from port_twin_app.models import Scenario, SimulationRun


class SimulationHistoryAPITests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.scenario = Scenario.objects.create(
            name="History API Test Scenario",
            description="Scenario used to test simulation history.",
            parameters={},
            status="ready",
        )

        cls.older_run = SimulationRun.objects.create(
            scenario=cls.scenario,
            status="completed",
        )

        cls.newer_run = SimulationRun.objects.create(
            scenario=cls.scenario,
            status="failed",
            error_message="Test simulation failure",
        )

    def setUp(self):
        self.client = APIClient()

    def test_history_returns_runs_newest_first(self):
        response = self.client.get("/api/v1/simulations/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["count"], 2)

        results = response.data["results"]

        self.assertEqual(
            results[0]["id"],
            str(self.newer_run.id),
        )
        self.assertEqual(
            results[1]["id"],
            str(self.older_run.id),
        )

    def test_history_includes_scenario_and_run_status(self):
        response = self.client.get("/api/v1/simulations/")

        self.assertEqual(response.status_code, 200)

        newer_run_data = response.data["results"][0]

        self.assertEqual(
            newer_run_data["scenario_name"],
            self.scenario.name,
        )
        self.assertEqual(newer_run_data["status"], "failed")
        self.assertEqual(
            newer_run_data["error_message"],
            "Test simulation failure",
        )
        self.assertIsNone(newer_run_data["summary"])
