import math
import re

from django.contrib.gis.geos import Point
from rest_framework import serializers

from port_twin_app.models import (
    Berth,
    MarineCondition,
    MarinePrediction,
    Scenario,
    SimulationRun,
    Vessel,
    VesselPosition,
)
from port_twin_app.services.marine_ai.contracts import (
    MarineForecastConfig,
    MarineGrid,
    MarinePredictionRequest,
)

SAFE_SIMULATION_ERROR_MESSAGE = (
    "Simulation execution failed. Check server logs for details."
)


def safe_simulation_error_message(simulation_run):
    if simulation_run.status == "failed":
        return SAFE_SIMULATION_ERROR_MESSAGE
    return ""


class APIErrorDetailSerializer(serializers.Serializer):
    code = serializers.CharField()
    message = serializers.CharField()
    details = serializers.JSONField()


class APIErrorResponseSerializer(serializers.Serializer):
    error = APIErrorDetailSerializer()


class VesselCreateRequestSerializer(serializers.ModelSerializer):
    class Meta:
        model = Vessel
        fields = [  # noqa: RUF012
            "name",
            "imo_number",
            "mmsi",
            "vessel_type",
            "length_overall",
            "width",
            "draft",
        ]

    def validate_imo_number(self, value):
        if not value:
            return value

        value = value.strip().upper()

        if value.startswith("IMO"):
            value = value[3:].strip()

        if not re.fullmatch(r"\d{7,8}", value):
            raise serializers.ValidationError(
                "Enter a valid IMO number, e.g. 'IMO 1234567'."
            )

        normalized_value = f"IMO {value}"

        if Vessel.objects.filter(imo_number=normalized_value).exists():
            raise serializers.ValidationError(
                "A vessel with this IMO number already exists."
            )

        return normalized_value


class VesselResponseSerializer(serializers.ModelSerializer):
    class Meta:
        model = Vessel
        fields = [  # noqa: RUF012
            "id",
            "name",
            "imo_number",
            "mmsi",
            "vessel_type",
            "length_overall",
            "width",
            "draft",
            "created_at",
            "updated_at",
        ]
        read_only_fields = fields


class VesselPositionQuerySerializer(serializers.Serializer):
    vessel = serializers.UUIDField(required=False)


class VesselListQuerySerializer(serializers.Serializer):
    page = serializers.IntegerField(
        required=False,
        min_value=1,
        default=1,
    )


class AssignedVesselSerializer(serializers.ModelSerializer):
    class Meta:
        model = Vessel
        fields = [  # noqa: RUF012
            "id",
            "name",
            "imo_number",
            "mmsi",
            "vessel_type",
        ]
        read_only_fields = fields


class BerthCreateRequestSerializer(serializers.ModelSerializer):
    longitude = serializers.FloatField(min_value=-180, max_value=180)
    latitude = serializers.FloatField(min_value=-90, max_value=90)

    max_vessel_length = serializers.DecimalField(
        max_digits=8,
        decimal_places=2,
        min_value=0.01,
    )
    max_vessel_draft = serializers.DecimalField(
        max_digits=8,
        decimal_places=2,
        min_value=0.01,
        required=False,
        allow_null=True,
    )

    class Meta:
        model = Berth
        fields = [  # noqa: RUF012
            "name",
            "longitude",
            "latitude",
            "max_vessel_length",
            "max_vessel_draft",
            "status",
        ]

    def validate_name(self, value):
        value = value.strip()

        if not value:
            raise serializers.ValidationError("Berth name cannot be empty.")

        if Berth.objects.filter(name__iexact=value).exists():
            raise serializers.ValidationError("A berth with this name already exists.")

        return value

    def create(self, validated_data):
        longitude = validated_data.pop("longitude")
        latitude = validated_data.pop("latitude")

        validated_data["location"] = Point(
            longitude,
            latitude,
            srid=4326,
        )

        return Berth.objects.create(**validated_data)


class BerthSerializer(serializers.ModelSerializer):
    longitude = serializers.SerializerMethodField()
    latitude = serializers.SerializerMethodField()
    is_occupied = serializers.SerializerMethodField()
    assigned_vessel = AssignedVesselSerializer(read_only=True, allow_null=True)

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
            "is_occupied",
            "assigned_vessel",
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

    def get_is_occupied(self, obj) -> bool:
        return hasattr(obj, "assigned_vessel")


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


class MarineConditionQuerySerializer(serializers.Serializer):
    source = serializers.ChoiceField(
        choices=MarineCondition.DataSource.choices,
        required=False,
    )


class PositiveFiniteFloatField(serializers.FloatField):
    def __init__(self, **kwargs):
        kwargs.setdefault("min_value", 0)
        super().__init__(**kwargs)

    def to_internal_value(self, data):
        if isinstance(data, bool):
            raise serializers.ValidationError("A numeric value is required.")

        value = super().to_internal_value(data)

        if not math.isfinite(value) or value <= 0:
            raise serializers.ValidationError(
                "Ensure this value is finite and greater than zero."
            )

        return value


class NonNegativeIntegerField(serializers.IntegerField):
    def to_internal_value(self, data):
        if isinstance(data, bool):
            raise serializers.ValidationError("A whole number is required.")

        return super().to_internal_value(data)


class ScenarioParametersSerializer(serializers.Serializer):
    simulation_hours = PositiveFiniteFloatField()

    expected_vessel_arrivals = NonNegativeIntegerField(
        min_value=0,
    )

    arrival_interval_hours = PositiveFiniteFloatField()

    average_service_time_hours = PositiveFiniteFloatField()

    available_berths = NonNegativeIntegerField(
        min_value=0,
        required=False,
        default=2,
    )

    def to_internal_value(self, data):
        if isinstance(data, dict):
            allowed_fields = set(self.fields)
            supplied_fields = set(data.keys())

            unknown_fields = supplied_fields - allowed_fields

            if unknown_fields:
                raise serializers.ValidationError(
                    {field: ["Unknown parameter."] for field in sorted(unknown_fields)}
                )

            required_fields = {
                "simulation_hours",
                "expected_vessel_arrivals",
                "arrival_interval_hours",
                "average_service_time_hours",
            }

            missing_fields = required_fields - supplied_fields

            if missing_fields:
                raise serializers.ValidationError(
                    {
                        field: ["This field is required."]
                        for field in sorted(missing_fields)
                    }
                )

        return super().to_internal_value(data)


class ScenarioCreateUpdateRequestSerializer(serializers.ModelSerializer):
    parameters = ScenarioParametersSerializer()

    class Meta:
        model = Scenario
        fields = [  # noqa: RUF012
            "name",
            "description",
            "parameters",
            "status",
        ]


class ScenarioResponseSerializer(serializers.ModelSerializer):
    class Meta:
        model = Scenario
        fields = [  # noqa: RUF012
            "id",
            "name",
            "description",
            "parameters",
            "status",
            "created_at",
            "updated_at",
        ]
        read_only_fields = fields


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
    reason = serializers.CharField(
        allow_null=True,
        required=False,
    )


class VesselResultsSerializer(serializers.Serializer):
    vessels = VesselSimulationResultSerializer(many=True)


class ScenarioSimulationResponseSerializer(serializers.Serializer):
    id = serializers.UUIDField()
    scenario_id = serializers.UUIDField()
    status = serializers.CharField()
    started_at = serializers.DateTimeField(allow_null=True)
    completed_at = serializers.DateTimeField(allow_null=True)
    summary = SimulationSummarySerializer(allow_null=True)


class QueueTimelinePointSerializer(serializers.Serializer):
    hour = serializers.FloatField()
    vessels_waiting = serializers.IntegerField()


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
    queue_timeline = QueueTimelinePointSerializer(
        many=True,
        allow_null=True,
    )


class SimulationErrorResponseSerializer(serializers.Serializer):
    id = serializers.UUIDField()
    scenario_id = serializers.UUIDField()
    status = serializers.CharField()
    error = serializers.CharField()


class SimulationHistoryItemSerializer(serializers.ModelSerializer):
    scenario_name = serializers.CharField(
        source="scenario.name",
        read_only=True,
    )
    summary = serializers.SerializerMethodField()
    error_message = serializers.SerializerMethodField()

    class Meta:
        model = SimulationRun
        fields = [  # noqa: RUF012
            "id",
            "scenario_id",
            "scenario_name",
            "status",
            "started_at",
            "completed_at",
            "error_message",
            "created_at",
            "summary",
        ]

    def get_summary(self, obj) -> dict | None:
        summary_result = next(
            (result for result in obj.results.all() if result.result_type == "summary"),
            None,
        )

        if summary_result is None:
            return None

        return SimulationSummarySerializer(summary_result.data).data

    def get_error_message(self, obj) -> str:
        return safe_simulation_error_message(obj)


class SimulationComparisonQuerySerializer(serializers.Serializer):
    run_a = serializers.UUIDField()
    run_b = serializers.UUIDField()

    def validate(self, attrs):
        if attrs["run_a"] == attrs["run_b"]:
            raise serializers.ValidationError(
                "Select two different simulation runs to compare."
            )

        return attrs


class SimulationComparisonRunSerializer(serializers.Serializer):
    id = serializers.UUIDField()
    scenario_name = serializers.CharField()
    summary = SimulationSummarySerializer()


class SimulationComparisonResponseSerializer(serializers.Serializer):
    run_a = SimulationComparisonRunSerializer()
    run_b = SimulationComparisonRunSerializer()
    differences = SimulationSummarySerializer()


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


class MarineGridInputSerializer(serializers.Serializer):
    west = serializers.FloatField()
    south = serializers.FloatField()
    east = serializers.FloatField()
    north = serializers.FloatField()
    nx = serializers.IntegerField(min_value=2)
    ny = serializers.IntegerField(min_value=2)
    crs = serializers.CharField(default="EPSG:4326")


class MarineForecastInputSerializer(serializers.Serializer):
    start_time = serializers.DateTimeField()


class MarinePredictionRequestSerializer(serializers.Serializer):
    grid = MarineGridInputSerializer()
    forecast = MarineForecastInputSerializer()
    bathymetry_m = serializers.JSONField()
    initial_water_level_m = serializers.JSONField()
    initial_current_u_ms = serializers.JSONField()
    initial_current_v_ms = serializers.JSONField()

    def validate(self, attrs):
        try:
            grid = MarineGrid(**attrs["grid"])

            forecast = MarineForecastConfig(
                start_time=attrs["forecast"]["start_time"].isoformat(),
            )

            prediction_request = MarinePredictionRequest(
                grid=grid,
                forecast=forecast,
                bathymetry_m=attrs["bathymetry_m"],
                initial_water_level_m=attrs["initial_water_level_m"],
                initial_current_u_ms=attrs["initial_current_u_ms"],
                initial_current_v_ms=attrs["initial_current_v_ms"],
            )
        except (TypeError, ValueError) as exc:
            raise serializers.ValidationError(str(exc)) from exc

        attrs["_prediction_request"] = prediction_request
        return attrs


class MarinePredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = MarinePrediction
        fields = [  # noqa: RUF012
            "id",
            "grid",
            "forecast",
            "input_data",
            "result_data",
            "model_version",
            "contract_version",
            "status",
            "error_message",
            "created_at",
        ]
        read_only_fields = fields


class DataSourceCountSerializer(serializers.Serializer):
    source = serializers.CharField()
    count = serializers.IntegerField()


class MarineDataProvenanceSerializer(serializers.Serializer):
    total_records = serializers.IntegerField()
    source_counts = DataSourceCountSerializer(many=True)
    latest_recorded_at = serializers.DateTimeField(allow_null=True)
    latest_ingested_at = serializers.DateTimeField(allow_null=True)


class VesselPositionDataStatusSerializer(serializers.Serializer):
    total_records = serializers.IntegerField()
    latest_recorded_at = serializers.DateTimeField(allow_null=True)
    latest_ingested_at = serializers.DateTimeField(allow_null=True)


class AIModelStatusSerializer(serializers.Serializer):
    backend = serializers.CharField()
    configured = serializers.BooleanField()
    readiness = serializers.CharField()
    configured_model_version = serializers.CharField(allow_blank=True)
    latest_prediction_status = serializers.CharField(allow_null=True)
    latest_prediction_model_version = serializers.CharField(allow_null=True)
    latest_prediction_contract_version = serializers.CharField(allow_null=True)
    latest_prediction_at = serializers.DateTimeField(allow_null=True)


class DataStatusSerializer(serializers.Serializer):
    generated_at = serializers.DateTimeField()
    marine_conditions = MarineDataProvenanceSerializer()
    vessel_positions = VesselPositionDataStatusSerializer()
    ai = AIModelStatusSerializer()
