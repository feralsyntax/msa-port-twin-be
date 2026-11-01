from .models import (
    Berth,
    MarineCondition,
    MarinePrediction,
    Scenario,
    SimulationRun,
    Vessel,
)

VESSEL_TYPE_CHOICES = Vessel.VesselType.choices
BERTH_STATUS_CHOICES = Berth.OperationalStatus.choices
MARINE_CONDITION_SOURCE_CHOICES = MarineCondition.DataSource.choices
SCENARIO_STATUS_CHOICES = Scenario.Status.choices
SIMULATION_RUN_STATUS_CHOICES = SimulationRun.Status.choices
MARINE_PREDICTION_STATUS_CHOICES = MarinePrediction.Status.choices
