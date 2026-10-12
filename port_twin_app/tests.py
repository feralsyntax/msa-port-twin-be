
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

        berth_data = response.data["results"][0]
        self.assertAlmostEqual(berth_data["longitude"], 39.65)
        self.assertAlmostEqual(berth_data["latitude"], -4.05)

    def test_retrieve_berth(self):
        response = self.client.get(
            reverse(
                "port_twin_app:berth-detail",
                kwargs={"pk": self.berth.pk},
            )
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["name"], self.berth.name)


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
            response.data["results"][0]["vessel"],
            str(self.vessel.pk),
        )


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
