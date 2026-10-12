from rest_framework import generics

from port_twin_app.models import Berth, MarineCondition, Vessel, VesselPosition
from port_twin_app.serializers import (
    BerthSerializer,
    MarineConditionSerializer,
    VesselPositionSerializer,
    VesselSerializer,
)


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
