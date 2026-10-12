from rest_framework import generics

from port_twin_app.models import Vessel
from port_twin_app.serializers import VesselSerializer


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
