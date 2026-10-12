from django.contrib.gis.geos import Point
from django.core.management.base import BaseCommand

from port_twin_app.models import Berth, Vessel


class Command(BaseCommand):
    help = "Populate the database with demonstration data for MSA Port Twin."

    def handle(self, *args, **options):
        self.seed_vessels()
        self.seed_berths()

        self.stdout.write(
            self.style.SUCCESS("Demo vessels and berths have been seeded.")
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
