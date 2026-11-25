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
            parameters={"simulation_hours": 24},
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
                        "service_time_hours": 8.0,
                        "service_order": 1,
                        "departure_hour": 8.0,
                        "waiting_hours": 0.0,
                        "status": "completed",
                        "reason": "No compatible berth was available.",
                    }
                ]
            },
        )

        SimulationResult.objects.create(
            simulation_run=cls.simulation_run,
            result_type="marine_conditions",
            data={
                "prediction_id": str(uuid.uuid4()),
                "model_version": "test-model",
                "contract_version": "2.0",
                "scope": "grid_level",
                "description": (
                    "Hourly summaries across the prediction grid. "
                    "These are not route-level conditions or navigational "
                    "safety assessments."
                ),
                "hourly_conditions": [
                    {
                        "step": step,
                        "start_time": f"2026-11-25T0{step}:00:00+00:00",
                        "end_time": f"2026-11-25T0{step + 1}:00:00+00:00",
                        "mean_water_level_m": 0.25,
                        "minimum_water_level_m": 0.1,
                        "maximum_water_level_m": 0.4,
                        "mean_current_speed_ms": 0.5,
                        "maximum_current_speed_ms": 0.5,
                    }
                    for step in range(1, 7)
                ],
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
        vessel = response.data["vessel_results"]["vessels"][0]

        self.assertEqual(vessel["service_time_hours"], 8.0)
        self.assertEqual(vessel["service_order"], 1)

        marine_conditions = response.data["marine_conditions"]

        self.assertIsNotNone(marine_conditions)
        self.assertEqual(marine_conditions["scope"], "grid_level")
        self.assertEqual(len(marine_conditions["hourly_conditions"]), 6)
        self.assertEqual(
            marine_conditions["prediction_id"],
            SimulationResult.objects.get(
                simulation_run=self.simulation_run,
                result_type="marine_conditions",
            ).data["prediction_id"],
        )

        self.assertEqual(
            response.data["queue_timeline"],
            [
                {"hour": 0.0, "vessels_waiting": 0},
                {"hour": 24.0, "vessels_waiting": 0},
            ],
        )

    def test_nonexistent_simulation_returns_404(self):
        response = self.client.get(f"/api/v1/simulations/{uuid.uuid4()}/")

        self.assertEqual(response.status_code, 404)

    def test_queue_timeline_tracks_overlapping_waiting_periods(self):
        vessel_result = SimulationResult.objects.get(
            simulation_run=self.simulation_run,
            result_type="vessel_results",
        )

        vessel_result.data = {
            "vessels": [
                {
                    "vessel_id": "vessel-1",
                    "arrival_hour": 0.0,
                    "service_start_hour": 4.0,
                    "status": "completed",
                },
                {
                    "vessel_id": "vessel-2",
                    "arrival_hour": 2.0,
                    "service_start_hour": 6.0,
                    "status": "completed",
                },
                {
                    "vessel_id": "vessel-3",
                    "arrival_hour": 4.0,
                    "service_start_hour": 4.0,
                    "status": "completed",
                },
                {
                    "vessel_id": "vessel-4",
                    "arrival_hour": 3.0,
                    "service_start_hour": None,
                    "status": "incompatible_berth",
                },
            ]
        }
        vessel_result.save(update_fields=["data"])

        response = self.client.get(self.url)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.data["queue_timeline"],
            [
                {"hour": 0.0, "vessels_waiting": 1},
                {"hour": 2.0, "vessels_waiting": 2},
                {"hour": 4.0, "vessels_waiting": 1},
                {"hour": 6.0, "vessels_waiting": 0},
                {"hour": 24.0, "vessels_waiting": 0},
            ],
        )
