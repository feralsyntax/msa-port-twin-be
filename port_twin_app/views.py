from rest_framework import generics

from port_twin_app.models import Berth, Vessel
from port_twin_app.serializers import BerthSerializer, VesselSerializer


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
