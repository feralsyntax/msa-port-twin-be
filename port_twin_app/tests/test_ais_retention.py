from datetime import timedelta

from django.contrib.gis.geos import Point
from django.test import TestCase
from django.utils import timezone

from port_twin_app.models import AISMessageReceipt, Vessel, VesselPosition
from port_twin_app.services.ais.retention import AISRetentionService


class AISRetentionServiceTests(TestCase):
    def setUp(self):
        self.now = timezone.now()
        self.vessel = Vessel.objects.create(name="Retention Test Vessel")

    def create_position(self, source, recorded_at):
        return VesselPosition.objects.create(
            vessel=self.vessel,
            source=source,
            position=Point(39.7, -4.05, srid=4326),
            recorded_at=recorded_at,
        )

    def test_deletes_expired_ais_positions(self):
        expired = self.create_position(
            VesselPosition.Source.AIS,
            self.now - timedelta(days=31),
        )
        recent = self.create_position(
            VesselPosition.Source.AIS,
            self.now - timedelta(days=1),
        )

        result = AISRetentionService(retention_days=30).cleanup(now=self.now)

        self.assertEqual(result["positions_deleted"], 1)
        self.assertFalse(VesselPosition.objects.filter(pk=expired.pk).exists())
        self.assertTrue(VesselPosition.objects.filter(pk=recent.pk).exists())

    def test_preserves_old_manual_positions(self):
        manual = self.create_position(
            VesselPosition.Source.MANUAL,
            self.now - timedelta(days=365),
        )

        AISRetentionService(retention_days=30).cleanup(now=self.now)

        self.assertTrue(VesselPosition.objects.filter(pk=manual.pk).exists())

    def test_deletes_expired_receipts(self):
        expired = AISMessageReceipt.objects.create(
            source="replay",
            fingerprint="a" * 64,
            received_at=self.now - timedelta(days=31),
        )
        recent = AISMessageReceipt.objects.create(
            source="replay",
            fingerprint="b" * 64,
            received_at=self.now - timedelta(days=1),
        )

        result = AISRetentionService(retention_days=30).cleanup(now=self.now)

        self.assertEqual(result["receipts_deleted"], 1)
        self.assertFalse(AISMessageReceipt.objects.filter(pk=expired.pk).exists())
        self.assertTrue(AISMessageReceipt.objects.filter(pk=recent.pk).exists())

    def test_rejects_non_positive_retention_period(self):
        with self.assertRaises(ValueError):
            AISRetentionService(retention_days=0)
