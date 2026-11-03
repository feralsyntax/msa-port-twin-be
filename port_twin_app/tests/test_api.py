
from datetime import timedelta

from django.contrib.gis.geos import Point
from django.urls import reverse
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APITestCase

from port_twin_app.models import (
    Berth,
    MarineCondition,
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
        self.assertTrue(
            Vessel.objects.filter(name="New Container Vessel").exists()
        )


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
        response = self.client.get(
            reverse("port_twin_app:vessel-position-list")
        )

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
        response = self.client.get(
            reverse("port_twin_app:marine-condition-list")
        )

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
