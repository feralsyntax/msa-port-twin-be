from rest_framework import serializers

from port_twin_app.models import Berth, Vessel, VesselPosition


class VesselSerializer(serializers.ModelSerializer):
    class Meta:
        model = Vessel
        fields = "__all__"


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
