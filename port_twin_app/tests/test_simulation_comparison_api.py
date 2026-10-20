import uuid

from django.test import TestCase
from rest_framework.test import APIClient

from port_twin_app.models import Scenario, SimulationResult, SimulationRun


class SimulationComparisonAPITests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.scenario = Scenario.objects.create(
            name="Comparison Test Scenario",
            description="Scenario used to test simulation comparisons.",
            parameters={},
            status="ready",
        )

        cls.run_a = SimulationRun.objects.create(
            scenario=cls.scenario,
            status="completed",
        )
        cls.run_b = SimulationRun.objects.create(
            scenario=cls.scenario,
            status="completed",
        )

        cls.summary_a = {
            "total_arrivals": 10,
            "vessels_served": 7,
            "vessels_waiting": 2,
            "vessels_in_service": 1,
            "average_waiting_hours": 5.0,
            "average_turnaround_hours": 9.0,
            "berth_utilization_percent": 60.0,
        }

        cls.summary_b = {
            "total_arrivals": 10,
            "vessels_served": 8,
            "vessels_waiting": 1,
            "vessels_in_service": 1,
            "average_waiting_hours": 3.0,
            "average_turnaround_hours": 7.0,
            "berth_utilization_percent": 70.0,
        }

        for simulation_run, summary in (
            (cls.run_a, cls.summary_a),
            (cls.run_b, cls.summary_b),
        ):
            SimulationResult.objects.create(
                simulation_run=simulation_run,
                result_type="summary",
                data=summary,
            )

    def setUp(self):
        self.client = APIClient()

    def test_compare_completed_runs(self):
        response = self.client.get(
            "/api/v1/simulations/compare/",
            {
                "run_a": str(self.run_a.id),
                "run_b": str(self.run_b.id),
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.data["run_a"]["id"],
            str(self.run_a.id),
        )
        self.assertEqual(
            response.data["run_b"]["id"],
            str(self.run_b.id),
        )
        self.assertEqual(
            response.data["differences"]["vessels_served"],
            1,
        )
        self.assertEqual(
            response.data["differences"]["average_waiting_hours"],
            -2.0,
        )
        self.assertEqual(
            response.data["differences"]["berth_utilization_percent"],
            10.0,
        )

    def test_same_run_cannot_be_compared_with_itself(self):
        response = self.client.get(
            "/api/v1/simulations/compare/",
            {
                "run_a": str(self.run_a.id),
                "run_b": str(self.run_a.id),
            },
        )

        self.assertEqual(response.status_code, 400)

    def test_nonexistent_run_returns_404(self):
        response = self.client.get(
            "/api/v1/simulations/compare/",
            {
                "run_a": str(self.run_a.id),
                "run_b": str(uuid.uuid4()),
            },
        )

        self.assertEqual(response.status_code, 404)
