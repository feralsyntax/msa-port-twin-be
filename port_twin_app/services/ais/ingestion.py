"""Persist decoded AIS messages using deduplication and vessel upserts."""

import hashlib
import json
import logging
from decimal import Decimal

from django.conf import settings
from django.contrib.gis.geos import Point
from django.db import transaction

from port_twin_app.models import AISMessageReceipt, Vessel, VesselPosition

logger = logging.getLogger(__name__)


class AISIngestionService:
    """Persist normalised AIS messages without overwriting missing data."""

    def __init__(self, source=None):
        self.source = source or settings.AIS_SOURCE
        self.dedup_window_seconds = settings.AIS_DEDUP_WINDOW_SECONDS

        if self.dedup_window_seconds <= 0:
            raise ValueError("AIS_DEDUP_WINDOW_SECONDS must be positive.")

    def ingest(self, messages):
        """Ingest a batch of decoded messages and return processing counts."""
        result = {
            "received": len(messages),
            "duplicates": 0,
            "vessels_created": 0,
            "positions_created": 0,
        }
        positions = []

        with transaction.atomic():
            for message in messages:
                fingerprint = self._fingerprint(message)

                _, created = AISMessageReceipt.objects.get_or_create(
                    source=self.source,
                    fingerprint=fingerprint,
                    defaults={"received_at": message.received_at},
                )

                if not created:
                    result["duplicates"] += 1
                    continue

                vessel = Vessel.objects.filter(mmsi=message.mmsi).first()

                # Static reports alone must not create unlocated vessels.
                if vessel is None and not message.has_position:
                    continue

                if vessel is None:
                    vessel = Vessel.objects.create(
                        mmsi=message.mmsi,
                        name=message.name or f"AIS vessel {message.mmsi}",
                        vessel_type=(message.vessel_type or Vessel.VesselType.UNKNOWN),
                    )
                    result["vessels_created"] += 1

                changed_fields = self._update_vessel(vessel, message)

                if changed_fields:
                    vessel.save(update_fields=changed_fields + ["updated_at"])

                if message.has_position:
                    positions.append(
                        VesselPosition(
                            vessel=vessel,
                            source=VesselPosition.Source.AIS,
                            position=Point(
                                message.longitude,
                                message.latitude,
                                srid=4326,
                            ),
                            speed_over_ground=self._decimal(message.speed_over_ground),
                            course_over_ground=self._decimal(
                                message.course_over_ground
                            ),
                            recorded_at=message.received_at,
                        )
                    )

            if positions:
                VesselPosition.objects.bulk_create(positions)
                result["positions_created"] = len(positions)

        return result

    def _fingerprint(self, message):
        """Build a stable fingerprint scoped to a configurable time bucket."""
        bucket = int(message.received_at.timestamp()) // self.dedup_window_seconds

        payload = {
            "mmsi": message.mmsi,
            "message_type": message.message_type,
            "bucket": bucket,
            "longitude": message.longitude,
            "latitude": message.latitude,
            "speed": message.speed_over_ground,
            "course": message.course_over_ground,
            "name": message.name,
            "imo": message.imo_number,
            "vessel_type": message.vessel_type,
            "length": message.length_overall,
            "width": message.width,
            "draft": message.draft,
        }

        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def _update_vessel(cls, vessel, message):
        """Update only valid fields supplied by the current AIS message."""
        changed_fields = []

        if message.name and vessel.name != message.name:
            vessel.name = message.name
            changed_fields.append("name")

        if message.vessel_type:
            valid_types = {choice for choice, _ in Vessel.VesselType.choices}
            if (
                message.vessel_type in valid_types
                and vessel.vessel_type != message.vessel_type
            ):
                vessel.vessel_type = message.vessel_type
                changed_fields.append("vessel_type")

        for field in ("length_overall", "width", "draft"):
            value = cls._decimal(getattr(message, field))
            if value is not None and getattr(vessel, field) != value:
                setattr(vessel, field, value)
                changed_fields.append(field)

        if message.imo_number:
            existing = (
                Vessel.objects.filter(imo_number=message.imo_number)
                .exclude(pk=vessel.pk)
                .exists()
            )

            if not existing and vessel.imo_number != message.imo_number:
                vessel.imo_number = message.imo_number
                changed_fields.append("imo_number")

        return changed_fields

    @staticmethod
    def _decimal(value):
        if value is None:
            return None
        return Decimal(str(value))
