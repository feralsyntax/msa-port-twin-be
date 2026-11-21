from datetime import UTC, datetime

from django.contrib.gis.geos import Point
from django.test import TestCase, override_settings

from port_twin_app.models import AISMessageReceipt, Vessel, VesselPosition
from port_twin_app.services.ais.decoder import DecodedAISMessage
from port_twin_app.services.ais.ingestion import AISIngestionService


@override_settings(
    AIS_SOURCE="test",
    AIS_DEDUP_WINDOW_SECONDS=10,
)
class AISIngestionTests(TestCase):
    def setUp(self):
        self.service = AISIngestionService()
        self.received_at = datetime(2026, 11, 21, 7, 0, tzinfo=UTC)

    def message(self, **overrides):
        values = {
            "mmsi": "123456789",
            "message_type": 1,
            "received_at": self.received_at,
            "longitude": 39.7,
            "latitude": -4.05,
            "speed_over_ground": 12.5,
            "course_over_ground": 180.0,
            "name": None,
            "imo_number": None,
            "vessel_type": None,
            "length_overall": None,
            "width": None,
            "draft": None,
        }
        values.update(overrides)
        return DecodedAISMessage(**values)

    def test_creates_vessel_and_ais_position(self):
        result = self.service.ingest([self.message()])

        self.assertEqual(result["vessels_created"], 1)
        self.assertEqual(result["positions_created"], 1)

        vessel = Vessel.objects.get(mmsi="123456789")
        position = VesselPosition.objects.get(vessel=vessel)

        self.assertEqual(position.source, VesselPosition.Source.AIS)
        self.assertEqual(position.position, Point(39.7, -4.05, srid=4326))

    def test_deduplicates_identical_message(self):
        message = self.message()

        first = self.service.ingest([message])
        second = self.service.ingest([message])

        self.assertEqual(first["positions_created"], 1)
        self.assertEqual(second["duplicates"], 1)
        self.assertEqual(VesselPosition.objects.count(), 1)
        self.assertEqual(AISMessageReceipt.objects.count(), 1)

    def test_static_message_does_not_create_unlocated_vessel(self):
        message = self.message(
            message_type=5,
            longitude=None,
            latitude=None,
            name="MV TEST",
            vessel_type="tanker",
        )

        self.service.ingest([message])

        self.assertFalse(Vessel.objects.filter(mmsi=message.mmsi).exists())
        self.assertEqual(VesselPosition.objects.count(), 0)

    def test_static_message_updates_existing_vessel(self):
        vessel = Vessel.objects.create(
            mmsi="123456789",
            name="Old name",
            vessel_type=Vessel.VesselType.UNKNOWN,
        )

        message = self.message(
            message_type=5,
            longitude=None,
            latitude=None,
            name="MV TEST",
            vessel_type="tanker",
            length_overall=180,
            width=30,
            draft=8.5,
        )

        self.service.ingest([message])
        vessel.refresh_from_db()

        self.assertEqual(vessel.name, "MV TEST")
        self.assertEqual(vessel.vessel_type, Vessel.VesselType.TANKER)
        self.assertEqual(float(vessel.length_overall), 180)
        self.assertEqual(float(vessel.width), 30)
        self.assertEqual(float(vessel.draft), 8.5)

    def test_missing_static_fields_do_not_overwrite_existing_values(self):
        vessel = Vessel.objects.create(
            mmsi="123456789",
            name="Existing vessel",
            vessel_type=Vessel.VesselType.TANKER,
            length_overall=180,
            width=30,
            draft=8,
        )

        self.service.ingest([self.message(name=None, vessel_type=None)])
        vessel.refresh_from_db()

        self.assertEqual(vessel.name, "Existing vessel")
        self.assertEqual(vessel.vessel_type, Vessel.VesselType.TANKER)
        self.assertEqual(float(vessel.length_overall), 180)
        self.assertEqual(float(vessel.width), 30)
        self.assertEqual(float(vessel.draft), 8)

    def test_preserves_existing_berth_assignment(self):
        from port_twin_app.models import Berth

        berth = Berth.objects.create(name="AIS Test Berth")
        vessel = Vessel.objects.create(
            mmsi="123456789",
            name="Existing vessel",
            berth=berth,
        )

        self.service.ingest([self.message(name="Updated vessel")])
        vessel.refresh_from_db()

        self.assertEqual(vessel.berth_id, berth.id)
        self.assertEqual(vessel.name, "Updated vessel")

    def test_identical_message_in_new_time_bucket_is_accepted(self):
        first = self.message()
        second = self.message(received_at=datetime(2026, 11, 21, 7, 0, 11, tzinfo=UTC))

        self.service.ingest([first])
        result = self.service.ingest([second])

        self.assertEqual(result["positions_created"], 1)
        self.assertEqual(VesselPosition.objects.count(), 2)
