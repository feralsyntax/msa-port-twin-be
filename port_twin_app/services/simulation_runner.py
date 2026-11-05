import math
from decimal import Decimal
from numbers import Real

from port_twin_app.models import Berth, Scenario, Vessel
from port_twin_app.services.simulation import (
    SimulationBerth,
    SimulationConfig,
    SimulationVessel,
    run_simulation,
)

REQUIRED_PARAMETERS = (
    "simulation_hours",
    "expected_vessel_arrivals",
    "arrival_interval_hours",
    "average_service_time_hours",
)


def _is_valid_dimension(value):
    """Check that a database dimension is a finite positive number."""

    return (
        isinstance(value, (Real, Decimal))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _validate_non_negative_integer(value, name):
    """Validate a non-negative integer configuration value."""

    if isinstance(value, int) and not isinstance(value, bool) and value < 0:
        raise ValueError(f"{name} cannot be negative.")

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a non-negative integer.")  # noqa: TRY004


def run_scenario_simulation(scenario: Scenario):
    """Load database records and execute a scenario simulation."""

    parameters = scenario.parameters

    if not isinstance(parameters, dict):
        raise ValueError("Scenario parameters must be a dictionary.")  # noqa: TRY004

    for parameter in REQUIRED_PARAMETERS:
        if parameter not in parameters:
            raise ValueError(f"Missing required scenario parameter: {parameter}.")

    config = SimulationConfig(
        duration_hours=parameters["simulation_hours"],
        expected_arrivals=parameters["expected_vessel_arrivals"],
        arrival_interval_hours=parameters["arrival_interval_hours"],
        service_time_hours=parameters["average_service_time_hours"],
    )

    _validate_non_negative_integer(
        config.expected_arrivals,
        "Expected arrivals",
    )

    available_berths = parameters.get("available_berths", 2)
    _validate_non_negative_integer(available_berths, "Available berths")

    # Load vessel profiles to use as templates for simulated arrivals.
    vessel_profiles = list(
        Vessel.objects.all().order_by("id")[: config.expected_arrivals]
    )

    if config.expected_arrivals > 0 and not vessel_profiles:
        raise ValueError(
            "At least one vessel profile is required to generate arrivals."
        )

    # Validate database vessel dimensions before creating simulation inputs.
    for profile in vessel_profiles:
        if not (
            _is_valid_dimension(profile.length_overall)
            and _is_valid_dimension(profile.draft)
        ):
            raise ValueError(
                "Vessel profiles must have valid length and draft dimensions."
            )

    # Generate a distinct simulation vessel for each expected arrival.
    # Reuse profile dimensions when fewer profiles than arrivals are available.
    vessels = [
        SimulationVessel(
            vessel_id=f"{profile.id}-arrival-{index + 1}",
            length=profile.length_overall,
            draft=profile.draft,
        )
        for index in range(config.expected_arrivals)
        for profile in [vessel_profiles[index % len(vessel_profiles)]]
    ]

    berth_records = list(
        Berth.objects.filter(status=Berth.OperationalStatus.ACTIVE).order_by("id")[
            :available_berths
        ]
    )

    # Validate active berth capacities before creating simulation inputs.
    for berth in berth_records:
        if not (
            _is_valid_dimension(berth.max_vessel_length)
            and _is_valid_dimension(berth.max_vessel_draft)
        ):
            raise ValueError(
                "Active berths must have valid length and draft capacity limits."
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
