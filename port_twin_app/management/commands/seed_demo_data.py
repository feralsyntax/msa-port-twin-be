from datetime import timedelta

from django.contrib.gis.geos import Point
from django.core.management.base import BaseCommand
from django.utils import timezone

from port_twin_app.models import Berth, MarineCondition, Vessel, VesselPosition


class Command(BaseCommand):
    help = "Populate the database with demonstration data for MSA Port Twin."

    def handle(self, *args, **options):
        self.seed_vessels()
        self.seed_berths()
        self.seed_vessel_positions()
        self.seed_marine_conditions()

        self.stdout.write(
            self.style.SUCCESS(
                "Demo vessels, berths, vessel positions, "
                "and marine conditions have been seeded."
            )
        )

    def seed_vessels(self):
        vessels = [
            {
                "name": "DEMO Container Vessel 01",
                "imo_number": "9000001",
                "mmsi": "600000001",
                "vessel_type": Vessel.VesselType.CONTAINER,
                "length_overall": 300.00,
                "width": 40.00,
                "draft": 12.00,
            },
            {
                "name": "DEMO Bulk Carrier 01",
                "imo_number": "9000002",
                "mmsi": "600000002",
                "vessel_type": Vessel.VesselType.BULK_CARRIER,
                "length_overall": 225.00,
                "width": 32.00,
                "draft": 10.50,
            },
            {
                "name": "DEMO Tanker 01",
                "imo_number": "9000003",
                "mmsi": "600000003",
                "vessel_type": Vessel.VesselType.TANKER,
                "length_overall": 190.00,
                "width": 30.00,
                "draft": 9.00,
            },
        ]

        for vessel_data in vessels:
            Vessel.objects.update_or_create(
                imo_number=vessel_data["imo_number"],
                defaults=vessel_data,
            )

    def seed_berths(self):
        berths = [
            {
                "name": "DEMO Berth 01",
                "location": Point(39.65, -4.05, srid=4326),
                "max_vessel_length": 330.00,
                "max_vessel_draft": 14.00,
                "status": Berth.OperationalStatus.ACTIVE,
            },
            {
                "name": "DEMO Berth 02",
                "location": Point(39.66, -4.04, srid=4326),
                "max_vessel_length": 250.00,
                "max_vessel_draft": 12.00,
                "status": Berth.OperationalStatus.ACTIVE,
            },
            {
                "name": "DEMO Berth 03",
                "location": Point(39.67, -4.03, srid=4326),
                "max_vessel_length": 200.00,
                "max_vessel_draft": 10.00,
                "status": Berth.OperationalStatus.MAINTENANCE,
            },
        ]

        for berth_data in berths:
            Berth.objects.update_or_create(
                name=berth_data["name"],
                defaults=berth_data,
            )

    def seed_vessel_positions(self):
        now = timezone.now()

        positions = [
            {
                "imo_number": "9000001",
                "longitude": 39.68,
                "latitude": -4.06,
                "speed": 4.5,
                "course": 135.0,
                "minutes_ago": 30,
            },
            {
                "imo_number": "9000001",
                "longitude": 39.67,
                "latitude": -4.07,
                "speed": 5.2,
                "course": 135.0,
                "minutes_ago": 15,
            },
            {
                "imo_number": "9000001",
                "longitude": 39.66,
                "latitude": -4.08,
                "speed": 3.8,
                "course": 135.0,
                "minutes_ago": 0,
            },
            {
                "imo_number": "9000002",
                "longitude": 39.70,
                "latitude": -4.03,
                "speed": 2.1,
                "course": 180.0,
                "minutes_ago": 0,
            },
            {
                "imo_number": "9000003",
                "longitude": 39.72,
                "latitude": -4.01,
                "speed": 0.0,
                "course": 90.0,
                "minutes_ago": 0,
            },
        ]

        for item in positions:
            vessel = Vessel.objects.get(imo_number=item["imo_number"])
            recorded_at = now - timedelta(minutes=item["minutes_ago"])

            VesselPosition.objects.update_or_create(
                vessel=vessel,
                recorded_at=recorded_at,
                defaults={
                    "position": Point(
                        item["longitude"],
                        item["latitude"],
                        srid=4326,
                    ),
                    "speed_over_ground": item["speed"],
                    "course_over_ground": item["course"],
                },
            )

    def seed_marine_conditions(self):
        now = timezone.now()

        conditions = [
            {
                "longitude": 39.66,
                "latitude": -4.05,
                "minutes_ago": 0,
                "water_level": 2.15,
                "current_speed": 0.65,
                "current_direction": 135.0,
                "wind_speed": 4.20,
            },
            {
                "longitude": 39.68,
                "latitude": -4.04,
                "minutes_ago": 30,
                "water_level": 2.08,
                "current_speed": 0.72,
                "current_direction": 140.0,
                "wind_speed": 4.50,
            },
            {
                "longitude": 39.70,
                "latitude": -4.03,
                "minutes_ago": 60,
                "water_level": 1.98,
                "current_speed": 0.80,
                "current_direction": 145.0,
                "wind_speed": 4.80,
            },
        ]

        for item in conditions:
            recorded_at = now - timedelta(minutes=item["minutes_ago"])

            MarineCondition.objects.update_or_create(
                location=Point(
                    item["longitude"],
                    item["latitude"],
                    srid=4326,
                ),
                recorded_at=recorded_at,
                source=MarineCondition.DataSource.DEMO,
                defaults={
                    "water_level": item["water_level"],
                    "current_speed": item["current_speed"],
                    "current_direction": item["current_direction"],
                    "wind_speed": item["wind_speed"],
                },
            )
