"""Estimate vessel arrival times from recent AIS tracks."""

import math
from dataclasses import dataclass
from datetime import timedelta

from django.utils import timezone

from port_twin_app.services.ais.tracking import (
    VesselZone,
    assess_vessel_track,
)
from port_twin_app.services.marine_data.domains.mombasa_channel import (
    PORT_OPERATIONS,
)


@dataclass(frozen=True)
class VesselETA:
    """Directional ETA to the port-operations boundary."""

    vessel_id: str
    eta_hours: float
    eta_at: object


def _time_to_rectangle_intersection(
    longitude,
    latitude,
    speed_knots,
    course_degrees,
    bounds,
):
    """Return hours until a projected track first intersects a rectangle."""

    latitude_radians = math.radians(latitude)
    longitude_scale = 60 * math.cos(latitude_radians)

    # Convert the rectangle into local nautical-mile coordinates,
    # relative to the vessel's latest position.
    x_min = (bounds.west - longitude) * longitude_scale
    x_max = (bounds.east - longitude) * longitude_scale
    y_min = (bounds.south - latitude) * 60
    y_max = (bounds.north - latitude) * 60

    course_radians = math.radians(course_degrees)

    # Course is measured clockwise from north.
    velocity_x = speed_knots * math.sin(course_radians)
    velocity_y = speed_knots * math.cos(course_radians)

    entry_time = -math.inf
    exit_time = math.inf

    for position, velocity, lower, upper in (
        (0.0, velocity_x, x_min, x_max),
        (0.0, velocity_y, y_min, y_max),
    ):
        if abs(velocity) < 1e-9:
            if not lower <= position <= upper:
                return None
            continue

        first = lower / velocity
        second = upper / velocity

        axis_entry = min(first, second)
        axis_exit = max(first, second)

        entry_time = max(entry_time, axis_entry)
        exit_time = min(exit_time, axis_exit)

        if entry_time > exit_time:
            return None

    if exit_time < 0:
        return None

    eta_hours = max(0.0, entry_time)

    return eta_hours


def estimate_vessel_eta(vessel, now=None):
    """Estimate a vessel's ETA to the port-operations boundary."""

    now = now or timezone.now()

    assessment = assess_vessel_track(vessel, now=now)

    if not assessment.is_reliable:
        return None

    if assessment.zone == VesselZone.PORT_OPERATIONS:
        return None

    latest = assessment.positions[0]

    if latest.position is None:
        return None

    speed = latest.speed_over_ground
    course = latest.course_over_ground

    if speed is None or course is None:
        return None

    try:
        speed = float(speed)
        course = float(course)
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(speed) or not math.isfinite(course):
        return None

    if speed <= 0 or not 0 <= course < 360:
        return None

    longitude = latest.position.x
    latitude = latest.position.y

    eta_hours = _time_to_rectangle_intersection(
        longitude=longitude,
        latitude=latitude,
        speed_knots=speed,
        course_degrees=course,
        bounds=PORT_OPERATIONS.bounds,
    )

    if eta_hours is None:
        return None

    observation_age_hours = (now - latest.recorded_at).total_seconds() / 3600

    remaining_eta_hours = eta_hours - observation_age_hours

    if remaining_eta_hours <= 0:
        return None

    return VesselETA(
        vessel_id=str(vessel.pk),
        eta_hours=remaining_eta_hours,
        eta_at=now + timedelta(hours=remaining_eta_hours),
    )
