"""Classify AIS vessel positions within the Mombasa monitoring area."""

from enum import StrEnum

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
