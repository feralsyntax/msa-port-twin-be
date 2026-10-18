from rest_framework import status
from rest_framework.test import APITestCase

from port_twin_app.models import Scenario


class ScenarioAPITests(APITestCase):
    def setUp(self):
        self.scenario = Scenario.objects.create(
            name="Standard Port Operations",
            description="Normal operating conditions.",
            parameters={"expected_arrivals": 5},
            status=Scenario.Status.DRAFT,
        )

    def test_list_scenarios(self):
        response = self.client.get("/api/v1/scenarios/")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        scenarios = response.data["results"]
        self.assertIn(
            str(self.scenario.id),
            [scenario["id"] for scenario in scenarios],
        )

    def test_create_scenario(self):
        payload = {
            "name": "High Traffic",
            "description": "Increased vessel arrivals.",
            "parameters": {"expected_arrivals": 10},
            "status": "ready",
        }

        response = self.client.post(
            "/api/v1/scenarios/",
            payload,
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(Scenario.objects.count(), 2)
        self.assertEqual(response.data["name"], "High Traffic")
        self.assertEqual(response.data["status"], "ready")

    def test_retrieve_scenario(self):
        response = self.client.get(f"/api/v1/scenarios/{self.scenario.id}/")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data["id"],
            str(self.scenario.id),
        )

    def test_partially_update_scenario(self):
        response = self.client.patch(
            f"/api/v1/scenarios/{self.scenario.id}/",
            {"name": "Updated Scenario"},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.scenario.refresh_from_db()
        self.assertEqual(self.scenario.name, "Updated Scenario")

    def test_parameters_must_be_a_json_object(self):
        response = self.client.post(
            "/api/v1/scenarios/",
            {
                "name": "Invalid Scenario",
                "parameters": ["not", "an", "object"],
            },
            format="json",
        )

        self.assertEqual(
            response.status_code,
            status.HTTP_400_BAD_REQUEST,
        )
        self.assertIn("parameters", response.data)
