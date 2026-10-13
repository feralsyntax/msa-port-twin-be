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

    if config.expected_arrivals < 0:
        raise ValueError("Expected arrivals cannot be negative.")

    # Load vessel profiles to use as templates for simulated arrivals.
    vessel_profiles = list(
        Vessel.objects.all().order_by("id")[: config.expected_arrivals]
    )

    if config.expected_arrivals > 0 and not vessel_profiles:
        raise ValueError(
            "At least one vessel profile is required to generate arrivals."
        )

    # Generate a distinct simulation vessel for each expected arrival.
    # Reuse profile dimensions when the database contains fewer profiles
    # than the scenario expects arrivals.
    vessels = [
        SimulationVessel(
            vessel_id=f"{profile.id}-arrival-{index + 1}",
            length=profile.length_overall,
            draft=profile.draft,
        )
        for index in range(config.expected_arrivals)
        for profile in [vessel_profiles[index % len(vessel_profiles)]]
    ]

    available_berths = parameters.get("available_berths", 2)

    if available_berths < 0:
        raise ValueError("Available berths cannot be negative.")

    berth_records = list(
        Berth.objects.filter(status="active").order_by("id")[:available_berths]
    )

    berths = [
        SimulationBerth(
            berth_id=str(berth.id),
            max_vessel_length=berth.max_vessel_length,
            max_vessel_draft=berth.max_vessel_draft,
        )
        for berth in berth_records
    ]

    return run_simulation(vessels, berths, config)
