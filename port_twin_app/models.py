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


class Berth(models.Model):
    """
    Stores port berths and their operational characteristics.

    A berth is a designated location in a port where a vessel can dock for loading, unloading, and other operations.

    Berth data will help simulate which vessels can be accommodated, assign vessels to suitable berths, and estimate waiting times.
    """

    class OperationalStatus(models.TextChoices):
        ACTIVE = "active", "Active"
        INACTIVE = "inactive", "Inactive"
        MAINTENANCE = "maintenance", "Under Maintenance"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100, unique=True)
    location = gis_models.PointField(geography=True, srid=4326, null=True, blank=True)
    max_vessel_length = models.DecimalField(
        max_digits=8,
        decimal_places=2,
        null=True,
        help_text="Maximum vessel length in metres.",
    )
    max_vessel_draft = models.DecimalField(
        max_digits=8,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Maximum permitted vessel draft in metres.",
    )
    status = models.CharField(
        max_length=20,
        choices=OperationalStatus.choices,
        default=OperationalStatus.ACTIVE,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]  # noqa: RUF012

    def __str__(self):
        return self.name


class MarineCondition(models.Model):
    """
    Stores observed and predicted marine conditions
    """

    class DataSource(models.TextChoices):
        OBSERVED = "observed", "Observed"
        AI_PREDICTION = "ai_prediction", "AI Prediction"
        EXTERNAL = "external", "External Data Source"
        DEMO = "demo", "Demonstration Data"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    location = gis_models.PointField(geography=True, srid=4326)
    recorded_at = models.DateTimeField(
        help_text="Time the conditions apply to or were observed."
    )
    water_level = models.DecimalField(
        max_digits=7,
        decimal_places=3,
        null=True,
        blank=True,
        help_text="Water level relative to a documented reference datum, in metres.",
    )
    current_speed = models.DecimalField(
        max_digits=6,
        decimal_places=3,
        null=True,
        blank=True,
        help_text="Current speed in metres per second.",
    )
    current_direction = models.DecimalField(
        max_digits=6,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Current direction in degrees.",
    )
    wind_speed = models.DecimalField(
        max_digits=6,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Wind speed in metres per second.",
    )
    source = models.CharField(
        max_length=20, choices=DataSource.choices, default=DataSource.DEMO
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-recorded_at"]  # noqa: RUF012
        indexes = [  # noqa: RUF012
            models.Index(fields=["recorded_at", "source"]),
        ]

    def __str__(self):
        return f"{self.source} marine condition at {self.recorded_at}"


class Scenario(models.Model):
    """
    Stores configuration for a port simulation scenario.

    A scenario represents a set of assumptions and parameters for a port simulation.

    It will help answer questions such as:
    - What happens to a vessel waiting times if arrivals increase?
    - How would congestion change if a berth becomes unavailable?
    - How might marine conditions affect vessel movement through the channel?
    """

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        READY = "ready", "Ready"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    parameters = models.JSONField(
        default=dict, help_text="Configuration values used by the simulation."
    )
    status = models.CharField(
        max_length=10, choices=Status.choices, default=Status.DRAFT
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]  # noqa: RUF012

    def __str__(self):
        return self.name


class SimulationRun(models.Model):
    """
    Tracks an individual execution of a simulation scenario.

    While a Scenario stores the configuration for a simulation, a SimulationRun represents one actual execution of that configuration.
    """

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    scenario = models.ForeignKey(
        Scenario, on_delete=models.PROTECT, related_name="runs"
    )
    status = models.CharField(
        max_length=10, choices=Status.choices, default=Status.PENDING
    )
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]  # noqa: RUF012
        indexes = [  # noqa: RUF012
            models.Index(fields=["status", "created_at"]),
        ]

    def __str__(self):
        return f"{self.scenario.name} - {self.status}"


class SimulationResult(models.Model):
    """
    Stores the outputs produced by a simulation run.

    It will give the front-end data to display vessel waiting times, queue lengths, berth allocations, and AI-generated marine predictions.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    simulation_run = models.ForeignKey(
        SimulationRun, on_delete=models.CASCADE, related_name="results"
    )
    result_type = models.CharField(
        max_length=50,
        help_text="Category of result, such as queue, berth allocation, or marine prediction.",
    )
    data = models.JSONField(
        default=dict, help_text="Structured output produced by the simulation."
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]  # noqa: RUF012
        indexes = [  # noqa: RUF012
            models.Index(fields=["simulation_run", "result_type"]),
        ]

    def __str__(self):
        return f"{self.result_type} - {self.simulation_run}"
