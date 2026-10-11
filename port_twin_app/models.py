import uuid

from django.db import models


class Vessel(models.Model):
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
