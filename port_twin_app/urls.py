from django.urls import path

from port_twin_app.views import (
    VesselDetailAPIView,
    VesselListAPIView,
)

app_name = "port_twin_app"

urlpatterns = [
    path("vessels/", VesselListAPIView.as_view(), name="vessel-list"),
    path(
        "vessels/<uuid:pk>/",
        VesselDetailAPIView.as_view(),
        name="vessel-detail",
    ),
]
