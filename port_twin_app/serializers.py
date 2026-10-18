import re

from rest_framework import serializers

from port_twin_app.models import (
    Berth,
    MarineCondition,
    Scenario,
    Vessel,
    VesselPosition,
)


class VesselSerializer(serializers.ModelSerializer):
    class Meta:
        model = Vessel
        fields = "__all__"

    def validate_imo_number(self, value):
        if not value:
            return value

        value = value.strip().upper()

        if value.startswith("IMO"):
            value = value[3:].strip()

        digits = value

        if not re.fullmatch(r"\d{7,8}", digits):
            raise serializers.ValidationError(
                "Enter a valid IMO number, e.g. 'IMO 1234567'."
            )

        return f"IMO {digits}"


class BerthSerializer(serializers.ModelSerializer):
    longitude = serializers.SerializerMethodField()
    latitude = serializers.SerializerMethodField()

    class Meta:
        model = Berth
        fields = [  # noqa: RUF012
            "id",
            "name",
            "longitude",
            "latitude",
            "max_vessel_length",
            "max_vessel_draft",
            "status",
            "created_at",
            "updated_at",
        ]
        read_only_fields = fields

    def get_longitude(self, obj) -> float | None:
        if obj.location is None:
            return None
        return obj.location.x

    def get_latitude(self, obj) -> float | None:
        if obj.location is None:
            return None
        return obj.location.y


class VesselPositionSerializer(serializers.ModelSerializer):
    longitude = serializers.SerializerMethodField()
    latitude = serializers.SerializerMethodField()

    class Meta:
        model = VesselPosition
        fields = [  # noqa: RUF012
            "id",
            "vessel",
            "longitude",
            "latitude",
            "speed_over_ground",
            "course_over_ground",
            "recorded_at",
            "created_at",
        ]
        read_only_fields = fields

    def get_longitude(self, obj) -> float:
        return obj.position.x

    def get_latitude(self, obj) -> float:
        return obj.position.y


class MarineConditionSerializer(serializers.ModelSerializer):
    longitude = serializers.SerializerMethodField()
    latitude = serializers.SerializerMethodField()

    class Meta:
        model = MarineCondition
        fields = [  # noqa: RUF012
            "id",
            "longitude",
            "latitude",
            "recorded_at",
            "water_level",
            "current_speed",
            "current_direction",
            "wind_speed",
            "source",
            "created_at",
        ]
        read_only_fields = fields

    def get_longitude(self, obj) -> float:
        return obj.location.x

    def get_latitude(self, obj) -> float:
        return obj.location.y


class ScenarioSerializer(serializers.ModelSerializer):
    class Meta:
        model = Scenario
        fields = "__all__"
        read_only_fields = [  # noqa: RUF012
            "id",
            "created_at",
            "updated_at",
        ]


class SimulationSummarySerializer(serializers.Serializer):
    total_arrivals = serializers.IntegerField()
    vessels_served = serializers.IntegerField()
    vessels_waiting = serializers.IntegerField()
    vessels_in_service = serializers.IntegerField()
    average_waiting_hours = serializers.FloatField()
    average_turnaround_hours = serializers.FloatField()
    berth_utilization_percent = serializers.FloatField()


class VesselSimulationResultSerializer(serializers.Serializer):
    vessel_id = serializers.CharField()
    arrival_hour = serializers.FloatField()
    berth_id = serializers.CharField(allow_null=True)
    service_start_hour = serializers.FloatField(allow_null=True)
    departure_hour = serializers.FloatField(allow_null=True)
    waiting_hours = serializers.FloatField(allow_null=True)
    status = serializers.CharField()


class VesselResultsSerializer(serializers.Serializer):
    vessels = VesselSimulationResultSerializer(many=True)


class ScenarioSimulationResponseSerializer(serializers.Serializer):
    id = serializers.UUIDField()
    scenario_id = serializers.UUIDField()
    status = serializers.CharField()
    started_at = serializers.DateTimeField(allow_null=True)
    completed_at = serializers.DateTimeField(allow_null=True)
    summary = SimulationSummarySerializer(allow_null=True)


class SimulationDetailResponseSerializer(serializers.Serializer):
    id = serializers.UUIDField()
    scenario_id = serializers.UUIDField()
    scenario_name = serializers.CharField()
    status = serializers.CharField()
    started_at = serializers.DateTimeField(allow_null=True)
    completed_at = serializers.DateTimeField(allow_null=True)
    error_message = serializers.CharField()
    created_at = serializers.DateTimeField()
    summary = SimulationSummarySerializer(allow_null=True)
    vessel_results = VesselResultsSerializer(allow_null=True)


class SimulationErrorResponseSerializer(serializers.Serializer):
    id = serializers.UUIDField()
    scenario_id = serializers.UUIDField()
    status = serializers.CharField()
    error = serializers.CharField()


class OperationsBerthOverviewSerializer(serializers.Serializer):
    total = serializers.IntegerField()
    active = serializers.IntegerField()
    inactive = serializers.IntegerField()
    maintenance = serializers.IntegerField()


class OperationsMetricsSerializer(serializers.Serializer):
    vessels_waiting = serializers.IntegerField()
    vessels_in_service = serializers.IntegerField()
    vessels_served = serializers.IntegerField()
    average_waiting_hours = serializers.FloatField()
    berth_utilization_percent = serializers.FloatField()


class OperationsDataStatusSerializer(serializers.Serializer):
    marine_condition_sources = serializers.ListField(
        child=serializers.CharField(),
    )
    latest_marine_condition_at = serializers.DateTimeField(
        allow_null=True,
    )


class OperationsOverviewSerializer(serializers.Serializer):
    generated_at = serializers.DateTimeField()
    total_vessels = serializers.IntegerField()
    berths = OperationsBerthOverviewSerializer()
    operations = OperationsMetricsSerializer(allow_null=True)
    latest_simulation_id = serializers.UUIDField(allow_null=True)
    latest_simulation_completed_at = serializers.DateTimeField(
        allow_null=True,
    )
    data_status = OperationsDataStatusSerializer()
