from port_twin_app.models import Berth, Scenario, Vessel
from port_twin_app.services.simulation import (
    SimulationBerth,
    SimulationConfig,
    SimulationVessel,
    run_simulation,
)


def run_scenario_simulation(scenario: Scenario):
    """Load database records and execute a scenario simulation."""

    parameters = scenario.parameters

    config = SimulationConfig(
        duration_hours=parameters["simulation_hours"],
        expected_arrivals=parameters["expected_vessel_arrivals"],
        arrival_interval_hours=parameters["arrival_interval_hours"],
        service_time_hours=parameters["average_service_time_hours"],
    )

    vessels = [
        SimulationVessel(
            vessel_id=str(vessel.id),
            length=vessel.length_overall,
            draft=vessel.draft,
        )
        for vessel in Vessel.objects.all().order_by("id")[: config.expected_arrivals]
    ]

    berths = [
        SimulationBerth(
            berth_id=str(berth.id),
            max_vessel_length=berth.max_vessel_length,
            max_vessel_draft=berth.max_vessel_draft,
        )
        for berth in Berth.objects.filter(status="active").order_by("id")[
            : parameters.get("available_berths", 2)
        ]
    ]

    return run_simulation(vessels, berths, config)
