from rest_framework import serializers

from port_twin_app.models import Berth, Vessel, VesselPosition


class VesselSerializer(serializers.ModelSerializer):
    class Meta:
        model = Vessel
        fields = "__all__"


class BerthSerializer(serializers.ModelSerializer):
    class Meta:
        model = Berth
        fields = "__all__"


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

    def get_longitude(self, obj):
        return obj.position.x

    def get_latitude(self, obj):
        return obj.position.y
