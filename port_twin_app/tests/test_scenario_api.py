from rest_framework import status
from rest_framework.test import APITestCase

from port_twin_app.models import Scenario


class ScenarioAPITests(APITestCase):
    def setUp(self):
        self.scenario = Scenario.objects.create(
            name="Standard Port Operations",
            description="Normal operating conditions.",
            parameters={
                "simulation_hours": 24,
                "expected_vessel_arrivals": 5,
                "arrival_interval_hours": 2,
                "average_service_time_hours": 8,
                "available_berths": 2,
            },
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
            "parameters": {
                "simulation_hours": 24,
                "expected_vessel_arrivals": 10,
                "arrival_interval_hours": 2,
                "average_service_time_hours": 8,
                "available_berths": 2,
            },
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
        self.assertIn("parameters", response.data["error"]["details"])


class ScenarioParameterConfigurationAPITests(APITestCase):
    def setUp(self):
        self.url = "/api/v1/scenarios/"

        self.base_payload = {
            "name": "Simulation Configuration Test",
            "parameters": {
                "simulation_hours": 24,
                "average_service_time_hours": 8,
            },
        }

    def create_scenario(self, parameters):
        payload = {
            **self.base_payload,
            "parameters": parameters,
        }

        return self.client.post(self.url, payload, format="json")

    def test_accepts_stochastic_arrival_configuration(self):
        response = self.create_scenario(
            {
                "simulation_hours": 24,
                "expected_vessel_arrivals": 12,
                "average_service_time_hours": 8,
                "arrival_source": "stochastic",
                "random_seed": 42,
            }
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

    def test_stochastic_arrivals_require_expected_arrivals(self):
        response = self.create_scenario(
            {
                "simulation_hours": 24,
                "average_service_time_hours": 8,
                "arrival_source": "stochastic",
            }
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn(
            "expected_vessel_arrivals",
            response.data["error"]["details"]["parameters"],
        )

    def test_ais_arrivals_do_not_require_expected_arrivals(self):
        response = self.create_scenario(
            {
                "simulation_hours": 24,
                "average_service_time_hours": 8,
                "arrival_source": "ais",
            }
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

    def test_accepts_uniform_service_time_configuration(self):
        response = self.create_scenario(
            {
                "simulation_hours": 24,
                "expected_vessel_arrivals": 5,
                "arrival_interval_hours": 2,
                "average_service_time_hours": 8,
                "service_time_distribution": "uniform",
                "minimum_service_time_hours": 4,
                "maximum_service_time_hours": 10,
            }
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

    def test_uniform_distribution_requires_both_bounds(self):
        response = self.create_scenario(
            {
                "simulation_hours": 24,
                "expected_vessel_arrivals": 5,
                "arrival_interval_hours": 2,
                "average_service_time_hours": 8,
                "service_time_distribution": "uniform",
                "minimum_service_time_hours": 4,
            }
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn(
            "maximum_service_time_hours",
            response.data["error"]["details"]["parameters"],
        )

    def test_rejects_maximum_service_time_below_minimum(self):
        response = self.create_scenario(
            {
                "simulation_hours": 24,
                "expected_vessel_arrivals": 5,
                "arrival_interval_hours": 2,
                "average_service_time_hours": 8,
                "service_time_distribution": "uniform",
                "minimum_service_time_hours": 10,
                "maximum_service_time_hours": 4,
            }
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_accepts_vessel_type_priorities(self):
        response = self.create_scenario(
            {
                "simulation_hours": 24,
                "expected_vessel_arrivals": 5,
                "arrival_interval_hours": 2,
                "average_service_time_hours": 8,
                "vessel_type_priorities": {
                    "container": 2,
                    "tanker": 0,
                    "bulk_carrier": 1,
                },
            }
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

    def test_rejects_invalid_priority_values(self):
        invalid_priorities = [
            {"container": -1},
            {"container": 1.5},
            {"container": True},
            {"": 1},
        ]

        for priorities in invalid_priorities:
            with self.subTest(priorities=priorities):
                response = self.create_scenario(
                    {
                        "simulation_hours": 24,
                        "expected_vessel_arrivals": 5,
                        "arrival_interval_hours": 2,
                        "average_service_time_hours": 8,
                        "vessel_type_priorities": priorities,
                    }
                )

                self.assertEqual(
                    response.status_code,
                    status.HTTP_400_BAD_REQUEST,
                )

    def test_rejects_negative_random_seed(self):
        response = self.create_scenario(
            {
                "simulation_hours": 24,
                "expected_vessel_arrivals": 5,
                "arrival_interval_hours": 2,
                "average_service_time_hours": 8,
                "random_seed": -1,
            }
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
