
from rest_framework import generics

from port_twin_app.models import Vessel
from port_twin_app.serializers import VesselSerializer


class VesselListAPIView(generics.ListAPIView):
    queryset = Vessel.objects.all()
    serializer_class = VesselSerializer


class VesselDetailAPIView(generics.RetrieveAPIView):
    queryset = Vessel.objects.all()
    serializer_class = VesselSerializer
