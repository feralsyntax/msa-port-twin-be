from django.conf import settings
from django.db.models import Max
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework import generics, status
from rest_framework.exceptions import APIException
from rest_framework.generics import GenericAPIView
from rest_framework.response import Response
from rest_framework.views import APIView

from port_twin_app.models import (
    Berth,
    MarineCondition,
    MarinePrediction,
    Scenario,
    SimulationRun,
    Vessel,
    VesselPosition,
)
from port_twin_app.serializers import (
    BerthSerializer,
    MarineConditionSerializer,
    MarinePredictionRequestSerializer,
    MarinePredictionSerializer,
    OperationsOverviewSerializer,
    ScenarioSerializer,
    ScenarioSimulationResponseSerializer,
    SimulationComparisonQuerySerializer,
    SimulationComparisonResponseSerializer,
    SimulationDetailResponseSerializer,
    SimulationHistoryItemSerializer,
    SimulationSummarySerializer,
    VesselPositionSerializer,
    VesselSerializer,
)
from port_twin_app.services.marine_ai.django_service import get_marine_inference_service
from port_twin_app.services.simulation_execution import execute_simulation_run


class VesselListAPIView(generics.ListAPIView):
    """
    Get a list of all vessels
    """

    queryset = Vessel.objects.all()
    serializer_class = VesselSerializer


class VesselDetailAPIView(generics.RetrieveAPIView):
    """
    Retrieve details of a single vessel identified by its UUID
    """

    queryset = Vessel.objects.all()
    serializer_class = VesselSerializer


class BerthListAPIView(generics.ListAPIView):
    """
    Get a list of all berths
    """

    queryset = Berth.objects.all()
    serializer_class = BerthSerializer


class BerthDetailAPIView(generics.RetrieveAPIView):
    """
    Retrieve details of a single berth identified by its UUID
    """

    queryset = Berth.objects.all()
    serializer_class = BerthSerializer


class VesselPositionListAPIView(generics.ListAPIView):
    """
    Retrieve either all vessel positions or the position of one vessel
    """

    serializer_class = VesselPositionSerializer

    def get_queryset(self):
        queryset = VesselPosition.objects.select_related("vessel")

        vessel_id = self.request.query_params.get("vessel")

        if vessel_id:
            queryset = queryset.filter(vessel_id=vessel_id)

        return queryset


class MarineConditionListAPIView(generics.ListAPIView):
    """
    Retrieve all marine-condition records and records filtered by source
    """

    serializer_class = MarineConditionSerializer

    def get_queryset(self):
        queryset = MarineCondition.objects.all()

        source = self.request.query_params.get("source")

        if source:
            queryset = queryset.filter(source=source)

        return queryset


class ScenarioListCreateAPIView(generics.ListCreateAPIView):
    """
    List saved scenarios or create a new scenario.
    """

    queryset = Scenario.objects.all()
    serializer_class = ScenarioSerializer


class ScenarioDetailAPIView(generics.RetrieveUpdateAPIView):
    """
    Retrieve or update a scenario identified by its UUID.
    """

    queryset = Scenario.objects.all()
    serializer_class = ScenarioSerializer


class ScenarioSimulationView(GenericAPIView):
    """
    Execute a simulation for a saved scenario.
    """

    serializer_class = ScenarioSimulationResponseSerializer

    def post(self, request, pk):
        scenario = get_object_or_404(Scenario, pk=pk)

        simulation_run = SimulationRun.objects.create(
            scenario=scenario,
        )

        try:
            execute_simulation_run(simulation_run)

        except ValueError as exc:
            simulation_run.refresh_from_db()

            return Response(
                {
                    "id": str(simulation_run.id),
                    "scenario_id": str(scenario.id),
                    "status": simulation_run.status,
                    "error": str(exc),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        simulation_run.refresh_from_db()

        summary = (
            simulation_run.results.filter(result_type="summary")
            .values_list("data", flat=True)
            .first()
        )

        return Response(
            {
                "id": str(simulation_run.id),
                "scenario_id": str(scenario.id),
                "status": simulation_run.status,
                "started_at": simulation_run.started_at,
                "completed_at": simulation_run.completed_at,
                "summary": summary,
            },
            status=status.HTTP_201_CREATED,
        )


class SimulationDetailView(APIView):
    """
    Retrieve a simulation run and its persisted results.
    """

    serializer_class = SimulationDetailResponseSerializer

    def get(self, request, pk):
        simulation_run = get_object_or_404(
            SimulationRun.objects.select_related("scenario"),
            pk=pk,
        )

        results = {
            result.result_type: result.data for result in simulation_run.results.all()
        }

        return Response(
            {
                "id": str(simulation_run.id),
                "scenario_id": str(simulation_run.scenario_id),
                "scenario_name": simulation_run.scenario.name,
                "status": simulation_run.status,
                "started_at": simulation_run.started_at,
                "completed_at": simulation_run.completed_at,
                "error_message": simulation_run.error_message,
                "created_at": simulation_run.created_at,
                "summary": results.get("summary"),
                "vessel_results": results.get("vessel_results"),
            },
            status=status.HTTP_200_OK,
        )


class SimulationHistoryListAPIView(generics.ListAPIView):
    serializer_class = SimulationHistoryItemSerializer

    queryset = (
        SimulationRun.objects.select_related("scenario")
        .prefetch_related("results")
        .all()
    )


class SimulationComparisonView(APIView):
    def get(self, request):
        query_serializer = SimulationComparisonQuerySerializer(
            data=request.query_params,
        )
        query_serializer.is_valid(raise_exception=True)

        run_a_id = query_serializer.validated_data["run_a"]
        run_b_id = query_serializer.validated_data["run_b"]

        run_a = get_object_or_404(
            SimulationRun.objects.select_related("scenario"),
            pk=run_a_id,
        )
        run_b = get_object_or_404(
            SimulationRun.objects.select_related("scenario"),
            pk=run_b_id,
        )

        for simulation_run in (run_a, run_b):
            if simulation_run.status != SimulationRun.Status.COMPLETED:
                return Response(
                    {
                        "detail": (
                            f"Simulation run {simulation_run.id} has not completed."
                        )
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

        summary_a_result = run_a.results.filter(
            result_type="summary",
        ).first()
        summary_b_result = run_b.results.filter(
            result_type="summary",
        ).first()

        if summary_a_result is None or summary_b_result is None:
            return Response(
                {"detail": "Both runs must have stored summary results."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        summary_a = SimulationSummarySerializer(
            summary_a_result.data,
        ).data
        summary_b = SimulationSummarySerializer(
            summary_b_result.data,
        ).data

        differences = {
            metric: summary_b[metric] - summary_a[metric] for metric in summary_a
        }

        response_data = {
            "run_a": {
                "id": run_a.id,
                "scenario_name": run_a.scenario.name,
                "summary": summary_a,
            },
            "run_b": {
                "id": run_b.id,
                "scenario_name": run_b.scenario.name,
                "summary": summary_b,
            },
            "differences": differences,
        }

        response_serializer = SimulationComparisonResponseSerializer(
            data=response_data,
        )
        response_serializer.is_valid(raise_exception=True)

        return Response(
            response_serializer.data,
            status=status.HTTP_200_OK,
        )


class OperationsOverviewView(GenericAPIView):
    """
    Return a consolidated overview of port operations.
    """

    serializer_class = OperationsOverviewSerializer

    def get(self, request):
        berths = Berth.objects.all()
        marine_conditions = MarineCondition.objects.all()

        latest_simulation = (
            SimulationRun.objects.filter(status="completed")
            .order_by("-completed_at", "-created_at")
            .first()
        )

        operations = None

        if latest_simulation:
            summary = (
                latest_simulation.results.filter(result_type="summary")
                .values_list("data", flat=True)
                .first()
            )

            if summary is not None:
                operations = {
                    "vessels_waiting": summary["vessels_waiting"],
                    "vessels_in_service": summary["vessels_in_service"],
                    "vessels_served": summary["vessels_served"],
                    "average_waiting_hours": (summary["average_waiting_hours"]),
                    "berth_utilization_percent": (summary["berth_utilization_percent"]),
                }

        data = {
            "generated_at": timezone.now(),
            "total_vessels": Vessel.objects.count(),
            "berths": {
                "total": berths.count(),
                "active": berths.filter(status="active").count(),
                "inactive": berths.filter(status="inactive").count(),
                "maintenance": berths.filter(status="maintenance").count(),
            },
            "operations": operations,
            "latest_simulation_id": (
                latest_simulation.id if latest_simulation else None
            ),
            "latest_simulation_completed_at": (
                latest_simulation.completed_at if latest_simulation else None
            ),
            "data_status": {
                "marine_condition_sources": sorted(
                    set(marine_conditions.values_list("source", flat=True))
                ),
                "latest_marine_condition_at": marine_conditions.aggregate(
                    latest=Max("recorded_at")
                )["latest"],
            },
        }

        serializer = self.get_serializer(data)
        return Response(serializer.data)


class MarinePredictionServiceUnavailable(APIException):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    default_detail = (
        "Marine prediction service is unavailable. Check the configured AI checkpoint."
    )
    default_code = "marine_prediction_unavailable"


class MarinePredictionCreateAPIView(APIView):
    """
    Generate and store a new forecast.
    """

    def post(self, request):
        request_serializer = MarinePredictionRequestSerializer(data=request.data)
        request_serializer.is_valid(raise_exception=True)

        prediction_request = request_serializer.validated_data["_prediction_request"]

        try:
            inference_service = get_marine_inference_service(
                settings.MARINE_AI_CHECKPOINT
            )
        except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
            raise MarinePredictionServiceUnavailable() from exc

        result = inference_service.predict(prediction_request)

        prediction = MarinePrediction.objects.create(
            grid=request.data["grid"],
            forecast=request.data["forecast"],
            input_data={
                key: value
                for key, value in request.data.items()
                if key not in {"grid", "forecast"}
            },
            result_data={
                "water_level_m": result.water_level_m,
                "current_u_ms": result.current_u_ms,
                "current_v_ms": result.current_v_ms,
                "status": result.status,
            },
            model_version=result.model_version,
            contract_version=result.contract_version,
            status=MarinePrediction.Status.COMPLETED,
        )

        response_serializer = MarinePredictionSerializer(prediction)

        return Response(
            response_serializer.data,
            status=status.HTTP_201_CREATED,
        )


class MarinePredictionDetailAPIView(generics.RetrieveAPIView):
    """Retrieve a previously stored marine forecast."""

    queryset = MarinePrediction.objects.all()
    serializer_class = MarinePredictionSerializer


class MarinePredictionListAPIView(generics.ListAPIView):
    """List stored marine forecasts."""

    queryset = MarinePrediction.objects.all()
    serializer_class = MarinePredictionSerializer
