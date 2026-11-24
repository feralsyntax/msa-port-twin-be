import math
from decimal import Decimal
from numbers import Real

from django.utils import timezone

from port_twin_app.models import Berth, Scenario, Vessel
from port_twin_app.services.ais.eta import estimate_vessel_eta
from port_twin_app.services.simulation import (
    SimulationBerth,
    SimulationConfig,
    SimulationVessel,
    run_simulation,
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

    arrival_source = parameters.get("arrival_source", "fixed")

    if arrival_source not in ("fixed", "ais"):
        raise ValueError("Arrival source must be 'fixed' or 'ais'.")

    required_parameters = {
        "simulation_hours",
        "average_service_time_hours",
    }

    if arrival_source == "fixed":
        required_parameters.update(
            {
                "expected_vessel_arrivals",
                "arrival_interval_hours",
            }
        )

    for parameter in required_parameters:
        if parameter not in parameters:
            raise ValueError(f"Missing required scenario parameter: {parameter}.")

    simulation_started_at = timezone.now()

    if arrival_source == "fixed":
        expected_arrivals = parameters["expected_vessel_arrivals"]
        arrival_interval = parameters["arrival_interval_hours"]
    else:
        expected_arrivals = 0
        arrival_interval = None

    config = SimulationConfig(
        duration_hours=parameters["simulation_hours"],
        expected_arrivals=expected_arrivals,
        arrival_interval_hours=arrival_interval,
        service_time_hours=parameters["average_service_time_hours"],
    )

    _validate_non_negative_integer(config.expected_arrivals, "Expected arrivals")

    available_berths = parameters.get("available_berths", 2)
    _validate_non_negative_integer(available_berths, "Available berths")

    if arrival_source == "fixed":
        # Load valid vessel profiles to use as templates for simulated arrivals.
        vessel_profiles = []

        if config.expected_arrivals > 0:
            for profile in Vessel.objects.all().order_by("id").iterator():
                if _is_valid_dimension(profile.length_overall) and _is_valid_dimension(
                    profile.draft
                ):
                    vessel_profiles.append(profile)

                    if len(vessel_profiles) >= config.expected_arrivals:
                        break

            if not vessel_profiles:
                raise ValueError(
                    "At least one vessel profile with valid length and draft "
                    "is required to generate arrivals."
                )

        # Generate a distinct simulation vessel for each expected arrival.
        # Reuse valid profile dimensions when fewer profiles than arrivals exist.
        vessels = [
            SimulationVessel(
                vessel_id=f"{profile.id}-arrival-{index + 1}",
                length=profile.length_overall,
                draft=profile.draft,
            )
            for index in range(config.expected_arrivals)
            for profile in [vessel_profiles[index % len(vessel_profiles)]]
        ]

    else:
        vessels_with_etas = []

        for vessel in Vessel.objects.all().order_by("id").iterator():
            if not (
                _is_valid_dimension(vessel.length_overall)
                and _is_valid_dimension(vessel.draft)
            ):
                continue

            eta = estimate_vessel_eta(
                vessel,
                now=simulation_started_at,
            )

            if eta is None or eta.eta_hours >= config.duration_hours:
                continue

            vessels_with_etas.append(
                (
                    eta.eta_hours,
                    SimulationVessel(
                        vessel_id=str(vessel.pk),
                        length=vessel.length_overall,
                        draft=vessel.draft,
                        arrival_hour=eta.eta_hours,
                    ),
                )
            )

        vessels_with_etas.sort(key=lambda item: (item[0], item[1].vessel_id))

        vessels = [item[1] for item in vessels_with_etas]

        config = SimulationConfig(
            duration_hours=config.duration_hours,
            expected_arrivals=len(vessels),
            arrival_interval_hours=None,
            service_time_hours=config.service_time_hours,
        )

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
