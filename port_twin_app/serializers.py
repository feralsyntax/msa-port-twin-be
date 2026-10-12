from rest_framework import serializers

from port_twin_app.models import Berth, Vessel


class VesselSerializer(serializers.ModelSerializer):
    class Meta:
        model = Vessel
        fields = "__all__"


class BerthSerializer(serializers.ModelSerializer):
    class Meta:
        model = Berth
        fields = "__all__"
