from django.urls import path

from port_twin_app import views

app_name = "port_twin_app"

urlpatterns = [
    path("vessels/", views.VesselListAPIView.as_view(), name="vessel-list"),
    path(
        "vessels/<uuid:pk>/",
        views.VesselDetailAPIView.as_view(),
        name="vessel-detail",
    ),
    path("berths/", views.BerthListAPIView.as_view(), name="berth-list"),
    path(
        "berths/<uuid:pk>/",
        views.BerthDetailAPIView.as_view(),
        name="berth-detail",
    ),
]
