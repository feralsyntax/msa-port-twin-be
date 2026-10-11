import uuid

from django.contrib.gis.db import models as gis_models
from django.db import models


class Vessel(models.Model):
    """
    Defines vessel identity and physical characteristics
    """

    class VesselType(models.TextChoices):
        CONTAINER = "container", "Container Ship"
        TANKER = "tanker", "Tanker"
        BULK_CARRIER = "bulk_carrier", "Bulk Carrier"
        GENERAL_CARGO = "general_cargo", "General Cargo"
        PASSENGER = "passenger", "Passenger"
        OTHER = "other", "Other"
        UNKNOWN = "unknown", "Unknown"

    id = models.UUIDField(
        primary_key=True, default=uuid.uuid4, unique=True, editable=False
    )
    name = models.CharField(max_length=255)
    # International Maritime Organization number
    imo_number = models.CharField(max_length=7, unique=True, blank=True, null=True)
    # Maritime Mobile Service Identity
    mmsi = models.CharField(max_length=9, unique=True, blank=True, null=True)
    vessel_type = models.CharField(
        max_length=20, choices=VesselType.choices, default=VesselType.UNKNOWN
    )
    length_overall = models.DecimalField(
        max_digits=8,
        decimal_places=2,
        blank=True,
        null=True,
        help_text="Vessel length in metres.",
    )
    width = models.DecimalField(
        max_digits=8,
        decimal_places=2,
        blank=True,
        null=True,
        help_text="Vessel width in metres.",
    )
    # Vessel draft is the vertical distance between the waterline and the lowest point of the vessel's hull
    draft = models.DecimalField(
        max_digits=8,
        decimal_places=2,
        blank=True,
        null=True,
        help_text="Vessel draft in metres.",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]  # noqa: RUF012

    def __str__(self):
        return self.name


class VesselPosition(models.Model):
    """
    Connects vessels to their geographical positions using PostGIS
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    vessel = models.ForeignKey(
        "Vessel", on_delete=models.CASCADE, related_name="position_history"
    )
    position = gis_models.PointField(geography=True, srid=4326)
    speed_over_ground = models.DecimalField(
        max_digits=6,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Vessel speed in knots.",
    )
    course_over_ground = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Vessel course in degrees.",
    )
    recorded_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-recorded_at"]  # noqa: RUF012
        indexes = [  # noqa: RUF012
            models.Index(fields=["vessel", "recorded_at"]),
        ]

    def __str__(self):
        return f"{self.vessel.name} at {self.recorded_at}"
