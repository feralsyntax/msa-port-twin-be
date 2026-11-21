"""Remove expired AIS position history and deduplication receipts."""

from datetime import timedelta

from django.conf import settings
from django.db import transaction
from django.utils import timezone

from port_twin_app.models import AISMessageReceipt, VesselPosition


class AISRetentionService:
    """Apply the configured retention period to AIS-specific records."""

    def __init__(self, retention_days=None):
        self.retention_days = (
            settings.AIS_POSITION_RETENTION_DAYS
            if retention_days is None
            else retention_days
        )

        if self.retention_days <= 0:
            raise ValueError("AIS retention days must be positive.")

    def cleanup(self, now=None):
        """Delete expired AIS records and return deletion counts."""
        cutoff = (now or timezone.now()) - timedelta(days=self.retention_days)

        with transaction.atomic():
            deleted_positions, _ = VesselPosition.objects.filter(
                source=VesselPosition.Source.AIS,
                recorded_at__lt=cutoff,
            ).delete()

            deleted_receipts, _ = AISMessageReceipt.objects.filter(
                received_at__lt=cutoff,
            ).delete()

        return {
            "cutoff": cutoff,
            "positions_deleted": deleted_positions,
            "receipts_deleted": deleted_receipts,
        }
