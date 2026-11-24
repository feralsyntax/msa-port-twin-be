"""Classify AIS vessel positions within the Mombasa monitoring area."""

from dataclasses import dataclass
from enum import StrEnum

from django.conf import settings
from django.utils import timezone

from port_twin_app.models import VesselPosition
from port_twin_app.services.marine_data.domains.mombasa_channel import (
    APPROACHES,
    PORT_OPERATIONS,
)


class VesselZone(StrEnum):
    UNKNOWN = "unknown"
    OUTSIDE_APPROACHES = "outside_approaches"
    APPROACHES = "approaches"
    PORT_OPERATIONS = "port_operations"


def classify_position(longitude, latitude):
    """Classify a WGS84 position against provisional monitoring envelopes."""
    if longitude is None or latitude is None:
        return VesselZone.UNKNOWN

    try:
        longitude = float(longitude)
        latitude = float(latitude)
    except (TypeError, ValueError, OverflowError):
        return VesselZone.UNKNOWN

    if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
        return VesselZone.UNKNOWN

    port_bounds = PORT_OPERATIONS.bounds

    if (
        port_bounds.west <= longitude <= port_bounds.east
        and port_bounds.south <= latitude <= port_bounds.north
    ):
        return VesselZone.PORT_OPERATIONS

    approach_bounds = APPROACHES.bounds

    if (
        approach_bounds.west <= longitude <= approach_bounds.east
        and approach_bounds.south <= latitude <= approach_bounds.north
    ):
        return VesselZone.APPROACHES

    return VesselZone.OUTSIDE_APPROACHES


@dataclass(frozen=True)
class TrackAssessment:
    """Assessment of a vessel's recent AIS position history."""

    vessel_id: str
    zone: VesselZone
    positions: tuple
    is_reliable: bool
    reason: str | None = None


def assess_vessel_track(vessel, now=None):
    """Retrieve and assess the two most recent AIS observations."""
    now = now or timezone.now()

    min_observations = settings.AIS_TRACK_MIN_OBSERVATIONS
    max_age = settings.AIS_TRACK_MAX_AGE_SECONDS
    min_interval = settings.AIS_TRACK_MIN_INTERVAL_SECONDS
    max_interval = settings.AIS_TRACK_MAX_INTERVAL_SECONDS

    positions = tuple(
        VesselPosition.objects.filter(
            vessel=vessel,
            source=VesselPosition.Source.AIS,
        ).order_by("-recorded_at", "-pk")[:min_observations]
    )

    if len(positions) < min_observations:
        return TrackAssessment(
            vessel_id=str(vessel.pk),
            zone=VesselZone.UNKNOWN,
            positions=positions,
            is_reliable=False,
            reason="insufficient_observations",
        )

    latest, previous = positions[0], positions[1]

    age = (now - latest.recorded_at).total_seconds()

    if age < 0 or age > max_age:
        return TrackAssessment(
            vessel_id=str(vessel.pk),
            zone=classify_position(
                latest.position.x,
                latest.position.y,
            ),
            positions=positions,
            is_reliable=False,
            reason="stale_observation",
        )

    interval = (latest.recorded_at - previous.recorded_at).total_seconds()

    if not min_interval <= interval <= max_interval:
        return TrackAssessment(
            vessel_id=str(vessel.pk),
            zone=classify_position(
                latest.position.x,
                latest.position.y,
            ),
            positions=positions,
            is_reliable=False,
            reason="invalid_observation_interval",
        )

    return TrackAssessment(
        vessel_id=str(vessel.pk),
        zone=classify_position(
            latest.position.x,
            latest.position.y,
        ),
        positions=positions,
        is_reliable=True,
    )
