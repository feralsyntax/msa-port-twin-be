"""Provisional nested computational domains for the Port of Mombasa.

Geographic bounds use WGS84 (EPSG:4326).
Projected grid calculations should use UTM Zone 37S (EPSG:32737).

The bounds are rectangular model extents, not navigational boundaries.
Grid spacing describes the target computational mesh, not source-data
resolution or guaranteed bathymetric accuracy.
"""

from dataclasses import dataclass
from math import isfinite

GEOGRAPHIC_CRS = "EPSG:4326"
PROJECTED_CRS = "EPSG:32737"

# Published chart extents used as initial geographic references.
# Chart 616: Approaches to Port Mombasa.
# Chart 666: Port Mombasa, including Kilindini and Port Reitz.
#
# These extents do not establish the exact limits of navigable water.
APPROACHES_CHART_REFERENCE = "BA Chart 616"
PORT_CHART_REFERENCE = "BA Chart 666"


@dataclass(frozen=True)
class GeographicBounds:
    west: float
    south: float
    east: float
    north: float

    def __post_init__(self):
        values = (self.west, self.south, self.east, self.north)

        if not all(isfinite(value) for value in values):
            raise ValueError("All geographic bounds must be finite.")

        if not -180 <= self.west <= 180:
            raise ValueError("West longitude must be between -180 and 180.")

        if not -180 <= self.east <= 180:
            raise ValueError("East longitude must be between -180 and 180.")

        if not -90 <= self.south <= 90:
            raise ValueError("South latitude must be between -90 and 90.")

        if not -90 <= self.north <= 90:
            raise ValueError("North latitude must be between -90 and 90.")

        if self.west >= self.east:
            raise ValueError("West longitude must be less than east longitude.")

        if self.south >= self.north:
            raise ValueError("South latitude must be less than north latitude.")

    def contains(self, other: "GeographicBounds") -> bool:
        """Return whether this extent fully contains another extent."""
        return (
            self.west <= other.west
            and self.south <= other.south
            and self.east >= other.east
            and self.north >= other.north
        )


@dataclass(frozen=True)
class GridZone:
    name: str
    bounds: GeographicBounds
    target_resolution_m: int
    parent: str | None
    source_reference: str

    def __post_init__(self):
        if self.target_resolution_m <= 0:
            raise ValueError("Target grid resolution must be positive.")


APPROACHES = GridZone(
    name="approaches",
    bounds=GeographicBounds(
        west=39.5500,
        south=-4.3333,
        east=39.8517,
        north=-3.8812,
    ),
    target_resolution_m=250,
    parent=None,
    source_reference=APPROACHES_CHART_REFERENCE,
)

PORT_OPERATIONS = GridZone(
    name="port_operations",
    bounds=GeographicBounds(
        west=39.5947,
        south=-4.1052,
        east=39.7187,
        north=-4.0317,
    ),
    target_resolution_m=100,
    parent=APPROACHES.name,
    source_reference=PORT_CHART_REFERENCE,
)

ZONES = (APPROACHES, PORT_OPERATIONS)
ZONES_BY_NAME = {zone.name: zone for zone in ZONES}


def validate_domain_zones() -> None:
    """Validate nesting and progressively finer grid spacing."""
    for zone in ZONES:
        if zone.parent is None:
            continue

        parent = ZONES_BY_NAME.get(zone.parent)
        if parent is None:
            raise ValueError(
                f"Zone '{zone.name}' references unknown parent '{zone.parent}'."
            )

        if not parent.bounds.contains(zone.bounds):
            raise ValueError(
                f"Zone '{zone.name}' extends outside parent '{parent.name}'."
            )

        if zone.target_resolution_m >= parent.target_resolution_m:
            raise ValueError(
                f"Zone '{zone.name}' must have finer target resolution "
                f"than parent '{parent.name}'."
            )


def get_zone(name: str) -> GridZone:
    """Return a configured zone by name."""
    try:
        return ZONES_BY_NAME[name]
    except KeyError:
        raise ValueError(f"Unknown Mombasa channel zone: {name}") from None


validate_domain_zones()
