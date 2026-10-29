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
        "scenarios/",
        views.ScenarioListCreateAPIView.as_view(),
        name="scenario-list-create",
    ),
    path(
        "scenarios/<uuid:pk>/",
        views.ScenarioDetailAPIView.as_view(),
        name="scenario-detail",
    ),
    path(
        "scenarios/<uuid:pk>/simulations/",
        views.ScenarioSimulationView.as_view(),
        name="scenario-simulations",
    ),
    path(
        "simulations/<uuid:pk>/",
        views.SimulationDetailView.as_view(),
        name="simulation-detail",
    ),
    path(
        "simulations/",
        views.SimulationHistoryListAPIView.as_view(),
        name="simulation-history",
    ),
    path(
        "simulations/compare/",
        views.SimulationComparisonView.as_view(),
        name="simulation-comparison",
    ),
    path(
        "overview/",
        views.OperationsOverviewView.as_view(),
        name="operations-overview",
    ),
    path(
        "data-status/",
        views.DataStatusAPIView.as_view(),
        name="data-status",
    ),
    path(
        "marine-predictions/",
        views.MarinePredictionListAPIView.as_view(),
        name="marine-prediction-list",
    ),
    path(
        "marine-predictions/create/",
        views.MarinePredictionCreateAPIView.as_view(),
        name="marine-prediction-create",
    ),
    path(
        "marine-predictions/<uuid:pk>/",
        views.MarinePredictionDetailAPIView.as_view(),
        name="marine-prediction-detail",
    ),
]
