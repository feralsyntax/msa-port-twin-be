import re

from rest_framework import serializers

from port_twin_app.models import (
    Berth,
    MarineCondition,
    Scenario,
    SimulationRun,
    Vessel,
    VesselPosition,
)

SAFE_SIMULATION_ERROR_MESSAGE = (
    "Simulation execution failed. Check server logs for details."
)


def safe_simulation_error_message(simulation_run):
    if simulation_run.status == "failed":
        return SAFE_SIMULATION_ERROR_MESSAGE
    return ""


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

    def validate_parameters(self, value):
        if not isinstance(value, dict):
            raise serializers.ValidationError("Parameters must be a JSON object.")

        return value


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

    def get_error_message(self, obj):
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


from port_twin_app.models import MarinePrediction
from port_twin_app.services.marine_ai.contracts import (
    MarineForecastConfig,
    MarineGrid,
    MarinePredictionRequest,
)


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
    steps = serializers.IntegerField(min_value=1)
    step_seconds = serializers.IntegerField(min_value=1)


class MarinePredictionRequestSerializer(serializers.Serializer):
    grid = MarineGridInputSerializer()
    forecast = MarineForecastInputSerializer()
    bathymetry_m = serializers.JSONField()
    initial_water_level_m = serializers.JSONField()
    wind_u_ms = serializers.JSONField()
    wind_v_ms = serializers.JSONField()
    boundary_conditions = serializers.JSONField()

    def validate(self, attrs):
        try:
            grid = MarineGrid(**attrs["grid"])

            forecast = MarineForecastConfig(
                start_time=attrs["forecast"]["start_time"].isoformat(),
                steps=attrs["forecast"]["steps"],
                step_seconds=attrs["forecast"]["step_seconds"],
            )

            prediction_request = MarinePredictionRequest(
                grid=grid,
                forecast=forecast,
                bathymetry_m=attrs["bathymetry_m"],
                initial_water_level_m=attrs["initial_water_level_m"],
                wind_u_ms=attrs["wind_u_ms"],
                wind_v_ms=attrs["wind_v_ms"],
                boundary_conditions=attrs["boundary_conditions"],
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
