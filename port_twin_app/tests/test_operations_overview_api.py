from django.test import TestCase
from rest_framework.test import APIClient

from port_twin_app.models import (
    Berth,
    MarineCondition,
    Scenario,
    SimulationResult,
    SimulationRun,
    Vessel,
)


class OperationsOverviewAPITests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.scenario = Scenario.objects.create(
            name="Overview Test Scenario",
            description="Scenario for overview API tests.",
            parameters={},
            status="ready",
        )

    def setUp(self):
        self.client = APIClient()
        self.url = "/api/v1/overview/"

    def test_overview_returns_database_and_simulation_metrics(self):
        Vessel.objects.create(
            name="Test Vessel",
            imo_number="IMO 1234567",
            mmsi="123456789",
            vessel_type="container",
            length_overall=200,
            width=30,
            draft=10,
        )

        Berth.objects.create(
            name="Active Test Berth",
            max_vessel_length=300,
            max_vessel_draft=15,
            status="active",
        )

        Berth.objects.create(
            name="Maintenance Test Berth",
            max_vessel_length=250,
            max_vessel_draft=12,
            status="maintenance",
        )

        MarineCondition.objects.create(
            location="SRID=4326;POINT (39.6682 -4.0435)",
            recorded_at="2026-10-17T08:00:00Z",
            water_level=1.2,
            current_speed=0.5,
            current_direction=90,
            wind_speed=10,
            source="demo",
        )

        simulation_run = SimulationRun.objects.create(
            scenario=self.scenario,
            status="completed",
        )

        SimulationResult.objects.create(
            simulation_run=simulation_run,
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

        response = self.client.get(self.url)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["total_vessels"], 1)
        self.assertEqual(response.data["berths"]["total"], 2)
        self.assertEqual(response.data["berths"]["active"], 1)
        self.assertEqual(response.data["berths"]["maintenance"], 1)
        self.assertEqual(response.data["operations"]["vessels_served"], 1)
        self.assertEqual(
            response.data["operations"]["berth_utilization_percent"],
            33.33,
        )
        self.assertEqual(
            response.data["data_status"]["marine_condition_sources"],
            ["demo"],
        )
        self.assertIsNotNone(response.data["latest_simulation_id"])

    def test_overview_without_completed_simulation(self):
        response = self.client.get(self.url)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["total_vessels"], 0)
        self.assertEqual(response.data["berths"]["total"], 0)
        self.assertIsNone(response.data["operations"])
        self.assertIsNone(response.data["latest_simulation_id"])
        self.assertIsNone(response.data["latest_simulation_completed_at"])
        self.assertEqual(
            response.data["data_status"]["marine_condition_sources"],
            [],
        )
        self.assertIsNone(response.data["data_status"]["latest_marine_condition_at"])
