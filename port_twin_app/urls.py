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
    path(
        "vessel-positions/",
        views.VesselPositionListAPIView.as_view(),
        name="vessel-position-list",
    ),
    path(
        "marine-conditions/",
        views.MarineConditionListAPIView.as_view(),
        name="marine-condition-list",
    ),
    path(
        "scenarios/<uuid:pk>/simulations/",
        views.ScenarioSimulationView.as_view(),
        name="scenario-simulations",
    ),
]
