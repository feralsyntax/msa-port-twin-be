from datetime import timedelta

from django.contrib.gis.geos import Point
from django.urls import reverse
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APITestCase

from port_twin_app.models import (
    Berth,
    MarineCondition,
    Scenario,
    Vessel,
    VesselPosition,
)


class VesselAPITests(APITestCase):
    def setUp(self):
        self.vessel = Vessel.objects.create(
            name="Test Container Vessel",
            imo_number="1234567",
            mmsi="123456789",
            vessel_type=Vessel.VesselType.CONTAINER,
        )

    def test_list_vessels(self):
        response = self.client.get(reverse("port_twin_app:vessel-list"))

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["count"], 1)
        self.assertEqual(response.data["results"][0]["name"], self.vessel.name)

    def test_retrieve_vessel(self):
        response = self.client.get(
            reverse(
                "port_twin_app:vessel-detail",
                kwargs={"pk": self.vessel.pk},
            )
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["id"], str(self.vessel.pk))

    def test_vessel_not_found(self):
        response = self.client.get(
            reverse(
                "port_twin_app:vessel-detail",
                kwargs={"pk": "00000000-0000-0000-0000-000000000000"},
            )
        )

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_register_vessel(self):
        payload = {
            "name": "New Container Vessel",
            "imo_number": "IMO 7654321",
            "mmsi": "987654321",
            "vessel_type": Vessel.VesselType.CONTAINER,
            "length_overall": "250.00",
            "width": "32.00",
            "draft": "12.00",
        }

        response = self.client.post(
            reverse("port_twin_app:vessel-list"),
            payload,
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data["name"], payload["name"])
        self.assertEqual(response.data["imo_number"], "IMO 7654321")
        self.assertEqual(response.data["mmsi"], payload["mmsi"])
        self.assertTrue(Vessel.objects.filter(name="New Container Vessel").exists())

    def test_register_vessel_normalizes_imo_number(self):
        payload = {
            "name": "IMO Normalization Test",
            "imo_number": "7654322",
        }

        response = self.client.post(
            reverse("port_twin_app:vessel-list"),
            payload,
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data["imo_number"], "IMO 7654322")

    def test_register_vessel_rejects_duplicate_imo(self):
        payload = {
            "name": "Duplicate IMO Vessel",
            "imo_number": "1234567",
        }

        response = self.client.post(
            reverse("port_twin_app:vessel-list"),
            payload,
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("imo_number", response.data)

    def test_register_vessel_rejects_duplicate_mmsi(self):
        payload = {
            "name": "Duplicate MMSI Vessel",
            "mmsi": "123456789",
        }

        response = self.client.post(
            reverse("port_twin_app:vessel-list"),
            payload,
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("mmsi", response.data)

    def test_register_vessel_requires_name(self):
        response = self.client.post(
            reverse("port_twin_app:vessel-list"),
            {"imo_number": "7654323"},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("name", response.data)


class BerthAPITests(APITestCase):
    def setUp(self):
        self.berth = Berth.objects.create(
            name="Test Berth",
            location=Point(39.65, -4.05, srid=4326),
            max_vessel_length=300,
            max_vessel_draft=14,
            status=Berth.OperationalStatus.ACTIVE,
        )

        self.vessel = Vessel.objects.create(
            name="MV Test Vessel",
            imo_number="1234567",
            mmsi="123456789",
            vessel_type=Vessel.VesselType.CONTAINER,
            berth=self.berth,
        )

    def test_list_berths_serializes_coordinates(self):
        response = self.client.get(reverse("port_twin_app:berth-list"))

        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.assertEqual(response.data["count"], 1)
        self.assertIsNone(response.data["next"])
        self.assertIsNone(response.data["previous"])

        berth_data = response.data["results"][0]

        self.assertEqual(berth_data["id"], str(self.berth.pk))
        self.assertEqual(berth_data["name"], self.berth.name)
        self.assertAlmostEqual(berth_data["longitude"], 39.65)
        self.assertAlmostEqual(berth_data["latitude"], -4.05)
        self.assertEqual(berth_data["max_vessel_length"], "300.00")
        self.assertEqual(berth_data["max_vessel_draft"], "14.00")
        self.assertEqual(berth_data["status"], Berth.OperationalStatus.ACTIVE)

    def test_retrieve_berth(self):
        response = self.client.get(
            reverse(
                "port_twin_app:berth-detail",
                kwargs={"pk": self.berth.pk},
            )
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)

        berth_data = response.data

        self.assertEqual(berth_data["id"], str(self.berth.pk))
        self.assertEqual(berth_data["name"], self.berth.name)
        self.assertAlmostEqual(berth_data["longitude"], 39.65)
        self.assertAlmostEqual(berth_data["latitude"], -4.05)
        self.assertEqual(berth_data["max_vessel_length"], "300.00")
        self.assertEqual(berth_data["max_vessel_draft"], "14.00")
        self.assertEqual(berth_data["status"], Berth.OperationalStatus.ACTIVE)

    def test_occupied_berth_returns_assigned_vessel(self):
        response = self.client.get(
            reverse(
                "port_twin_app:berth-detail",
                kwargs={"pk": self.berth.pk},
            )
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data["is_occupied"])
        self.assertEqual(
            response.data["assigned_vessel"]["id"],
            str(self.vessel.id),
        )
        self.assertEqual(
            response.data["assigned_vessel"]["name"],
            self.vessel.name,
        )
        self.assertEqual(
            response.data["assigned_vessel"]["imo_number"],
            self.vessel.imo_number,
        )
        self.assertEqual(
            response.data["assigned_vessel"]["mmsi"],
            self.vessel.mmsi,
        )
        self.assertEqual(
            response.data["assigned_vessel"]["vessel_type"],
            self.vessel.vessel_type,
        )

    def test_unoccupied_berth_returns_null_assigned_vessel(self):
        self.vessel.delete()

        response = self.client.get(
            reverse(
                "port_twin_app:berth-detail",
                kwargs={"pk": self.berth.pk},
            )
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertFalse(response.data["is_occupied"])
        self.assertIsNone(response.data["assigned_vessel"])

    def test_list_berths_includes_occupancy_information(self):
        response = self.client.get(reverse("port_twin_app:berth-list"))

        self.assertEqual(response.status_code, status.HTTP_200_OK)

        berth_data = response.data["results"][0]

        self.assertTrue(berth_data["is_occupied"])
        self.assertEqual(
            berth_data["assigned_vessel"]["id"],
            str(self.vessel.id),
        )

    def test_inactive_berth_can_still_be_occupied(self):
        self.berth.status = Berth.OperationalStatus.INACTIVE
        self.berth.save()

        response = self.client.get(
            reverse(
                "port_twin_app:berth-detail",
                kwargs={"pk": self.berth.pk},
            )
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["status"], Berth.OperationalStatus.INACTIVE)
        self.assertTrue(response.data["is_occupied"])
        self.assertIsNotNone(response.data["assigned_vessel"])


class VesselPositionAPITests(APITestCase):
    def setUp(self):
        self.vessel = Vessel.objects.create(
            name="Test Vessel",
            imo_number="2345678",
        )
        self.other_vessel = Vessel.objects.create(
            name="Another Test Vessel",
            imo_number="3456789",
        )

        now = timezone.now()

        self.position = VesselPosition.objects.create(
            vessel=self.vessel,
            position=Point(39.66, -4.08, srid=4326),
            speed_over_ground=3.8,
            course_over_ground=135,
            recorded_at=now,
        )

        VesselPosition.objects.create(
            vessel=self.other_vessel,
            position=Point(39.70, -4.03, srid=4326),
            recorded_at=now - timedelta(minutes=5),
        )

    def test_position_list_serializes_coordinates(self):
        response = self.client.get(reverse("port_twin_app:vessel-position-list"))

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["count"], 2)

        position_data = next(
            item
            for item in response.data["results"]
            if item["id"] == str(self.position.pk)
        )
        self.assertAlmostEqual(position_data["longitude"], 39.66)
        self.assertAlmostEqual(position_data["latitude"], -4.08)

    def test_filter_positions_by_vessel(self):
        response = self.client.get(
            reverse("port_twin_app:vessel-position-list"),
            {"vessel": str(self.vessel.pk)},
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["count"], 1)
        self.assertEqual(
            str(response.data["results"][0]["vessel"]),
            str(self.vessel.pk),
        )

    def test_retrieve_vessel_position_history(self):
        response = self.client.get(
            reverse(
                "port_twin_app:vessel-position-history",
                kwargs={"vessel_id": self.vessel.pk},
            )
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["count"], 1)

        position_data = response.data["results"][0]

        self.assertAlmostEqual(position_data["longitude"], 39.66)
        self.assertAlmostEqual(position_data["latitude"], -4.08)
        self.assertEqual(
            float(position_data["speed_over_ground"]),
            3.8,
        )
        self.assertEqual(
            float(position_data["course_over_ground"]),
            135,
        )

    def test_retrieve_vessel_position_history_empty(self):
        vessel_without_positions = Vessel.objects.create(
            name="Vessel Without Positions",
            imo_number="4567890",
        )

        response = self.client.get(
            reverse(
                "port_twin_app:vessel-position-history",
                kwargs={"vessel_id": vessel_without_positions.pk},
            )
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["count"], 0)
        self.assertEqual(response.data["results"], [])

    def test_retrieve_vessel_position_history_vessel_not_found(self):
        response = self.client.get(
            reverse(
                "port_twin_app:vessel-position-history",
                kwargs={
                    "vessel_id": "00000000-0000-0000-0000-000000000000",
                },
            )
        )

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)


class MarineConditionAPITests(APITestCase):
    def setUp(self):
        self.condition = MarineCondition.objects.create(
            location=Point(39.66, -4.05, srid=4326),
            recorded_at=timezone.now(),
            water_level=2.15,
            current_speed=0.65,
            current_direction=135,
            wind_speed=4.2,
            source=MarineCondition.DataSource.DEMO,
        )

        MarineCondition.objects.create(
            location=Point(39.68, -4.04, srid=4326),
            recorded_at=timezone.now() - timedelta(minutes=10),
            source=MarineCondition.DataSource.AI_PREDICTION,
        )

    def test_list_conditions_serializes_coordinates(self):
        response = self.client.get(reverse("port_twin_app:marine-condition-list"))

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["count"], 2)

        condition_data = next(
            item
            for item in response.data["results"]
            if item["id"] == str(self.condition.pk)
        )
        self.assertAlmostEqual(condition_data["longitude"], 39.66)
        self.assertAlmostEqual(condition_data["latitude"], -4.05)

    def test_filter_conditions_by_source(self):
        response = self.client.get(
            reverse("port_twin_app:marine-condition-list"),
            {"source": MarineCondition.DataSource.DEMO},
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["count"], 1)
        self.assertEqual(
            response.data["results"][0]["source"],
            MarineCondition.DataSource.DEMO,
        )


class ScenarioAPITests(APITestCase):
    def setUp(self):
        self.valid_parameters = {
            "simulation_hours": 24,
            "expected_vessel_arrivals": 5,
            "arrival_interval_hours": 2,
            "average_service_time_hours": 8,
            "available_berths": 2,
        }

        self.scenario = Scenario.objects.create(
            name="Test Scenario",
            description="Scenario for API testing",
            parameters=self.valid_parameters,
            status=Scenario.Status.DRAFT,
        )

    def test_list_scenarios(self):
        response = self.client.get(reverse("port_twin_app:scenario-list-create"))

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["results"][0]["name"], self.scenario.name)

    def test_retrieve_scenario(self):
        response = self.client.get(
            reverse(
                "port_twin_app:scenario-detail",
                kwargs={"pk": self.scenario.pk},
            )
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["id"], str(self.scenario.pk))
        self.assertEqual(response.data["parameters"], self.valid_parameters)

    def test_create_scenario(self):
        payload = {
            "name": "New Scenario",
            "description": "New simulation configuration",
            "parameters": self.valid_parameters,
            "status": Scenario.Status.READY,
        }

        response = self.client.post(
            reverse("port_twin_app:scenario-list-create"),
            payload,
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data["name"], payload["name"])
        self.assertEqual(response.data["parameters"], self.valid_parameters)
        self.assertTrue(Scenario.objects.filter(name="New Scenario").exists())

    def test_create_scenario_applies_default_available_berths(self):
        payload = {
            "name": "Default Berths Scenario",
            "parameters": {
                "simulation_hours": 24,
                "expected_vessel_arrivals": 5,
                "arrival_interval_hours": 2,
                "average_service_time_hours": 8,
            },
        }

        response = self.client.post(
            reverse("port_twin_app:scenario-list-create"),
            payload,
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data["parameters"]["available_berths"], 2)

    def test_update_scenario(self):
        response = self.client.patch(
            reverse(
                "port_twin_app:scenario-detail",
                kwargs={"pk": self.scenario.pk},
            ),
            {"name": "Updated Scenario"},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.scenario.refresh_from_db()
        self.assertEqual(self.scenario.name, "Updated Scenario")

    def test_create_scenario_rejects_invalid_parameters(self):
        invalid_parameters = [
            {"simulation_hours": 0},
            {"expected_vessel_arrivals": -1},
            {"arrival_interval_hours": 0},
            {"average_service_time_hours": -1},
            {"available_berths": -1},
        ]

        for invalid_field in invalid_parameters:
            with self.subTest(invalid_field=invalid_field):
                parameters = {
                    **self.valid_parameters,
                    **invalid_field,
                }

                response = self.client.post(
                    reverse("port_twin_app:scenario-list-create"),
                    {
                        "name": "Invalid Scenario",
                        "parameters": parameters,
                    },
                    format="json",
                )

                self.assertEqual(
                    response.status_code,
                    status.HTTP_400_BAD_REQUEST,
                )
