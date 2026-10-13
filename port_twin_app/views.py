from django.shortcuts import get_object_or_404
from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.views import APIView

from port_twin_app.models import (
    Berth,
    MarineCondition,
    Scenario,
    SimulationRun,
    Vessel,
    VesselPosition,
)
from port_twin_app.serializers import (
    BerthSerializer,
    MarineConditionSerializer,
    VesselPositionSerializer,
    VesselSerializer,
)
from port_twin_app.services.simulation_execution import execute_simulation_run


class VesselListAPIView(generics.ListAPIView):
    """
    Returns a list of all vessels
    """

    queryset = Vessel.objects.all()
    serializer_class = VesselSerializer


class VesselDetailAPIView(generics.RetrieveAPIView):
    """
    Returns details of a single vessel identified by its UUID
    """

    queryset = Vessel.objects.all()
    serializer_class = VesselSerializer


class BerthListAPIView(generics.ListAPIView):
    """
    Returns a list of all berths
    """

    queryset = Berth.objects.all()
    serializer_class = BerthSerializer


class BerthDetailAPIView(generics.RetrieveAPIView):
    """
    Returns details of a single berth identified by its UUID
    """

    queryset = Berth.objects.all()
    serializer_class = BerthSerializer


class VesselPositionListAPIView(generics.ListAPIView):
    """
    Returns either all vessel positions or the position of one vessel
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
    Returns all marine-condition records and records filtered by source
    """

    serializer_class = MarineConditionSerializer

    def get_queryset(self):
        queryset = MarineCondition.objects.all()

        source = self.request.query_params.get("source")

        if source:
            queryset = queryset.filter(source=source)

        return queryset


class ScenarioSimulationView(APIView):
    """
    Executes a simulation for a saved scenario.
    """

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
