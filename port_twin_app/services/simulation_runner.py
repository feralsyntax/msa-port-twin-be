import math
import random
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

DEFAULT_RANDOM_SEED = 42


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


def _load_valid_vessel_profiles():
    """Load vessel profiles with valid physical dimensions."""

    return [
        vessel
        for vessel in Vessel.objects.all().order_by("id").iterator()
        if _is_valid_dimension(vessel.length_overall)
        and _is_valid_dimension(vessel.draft)
    ]


def _get_service_time(parameters, rng):
    """Return a sampled service duration, or None for the fixed default."""

    if parameters.get("service_time_distribution", "fixed") == "fixed":
        return None

    minimum = parameters["minimum_service_time_hours"]
    maximum = parameters["maximum_service_time_hours"]

    return rng.uniform(minimum, maximum)


def _build_simulation_vessel(
    profile,
    vessel_id,
    parameters,
    rng,
    arrival_hour=None,
):
    """Build a simulation vessel from a database vessel profile."""

    return SimulationVessel(
        vessel_id=str(vessel_id),
        length=profile.length_overall,
        draft=profile.draft,
        arrival_hour=arrival_hour,
        vessel_type=profile.vessel_type or Vessel.VesselType.UNKNOWN,
        service_time_hours=_get_service_time(parameters, rng),
    )


def _generate_stochastic_vessels(parameters, duration_hours, rng):
    """Generate vessel arrivals using a Poisson process."""

    expected_arrivals = parameters["expected_vessel_arrivals"]

    if expected_arrivals == 0:
        return []

    arrival_rate = expected_arrivals / duration_hours
    profiles = _load_valid_vessel_profiles()

    if not profiles:
        raise ValueError(
            "At least one vessel profile with valid length and draft "
            "is required to generate arrivals."
        )

    vessels = []
    arrival_hour = 0.0
    arrival_number = 0

    while True:
        arrival_hour += rng.expovariate(arrival_rate)

        if arrival_hour >= duration_hours:
            break

        profile = rng.choice(profiles)
        arrival_number += 1

        vessels.append(
            _build_simulation_vessel(
                profile=profile,
                vessel_id=f"{profile.id}-arrival-{arrival_number}",
                parameters=parameters,
                rng=rng,
                arrival_hour=arrival_hour,
            )
        )

    return vessels


def run_scenario_simulation(scenario: Scenario):
    """Load database records and execute a scenario simulation."""

    parameters = scenario.parameters

    if not isinstance(parameters, dict):
        raise ValueError("Scenario parameters must be a dictionary.")  # noqa: TRY004

    arrival_source = parameters.get("arrival_source", "fixed")

    if arrival_source not in ("fixed", "ais", "stochastic"):
        raise ValueError("Arrival source must be 'fixed', 'ais', or 'stochastic'.")

    service_distribution = parameters.get("service_time_distribution", "fixed")

    if service_distribution not in ("fixed", "uniform"):
        raise ValueError("Service time distribution must be 'fixed' or 'uniform'.")

    required_parameters = {
        "simulation_hours",
        "average_service_time_hours",
    }

    if arrival_source in ("fixed", "stochastic"):
        required_parameters.add("expected_vessel_arrivals")

    if arrival_source == "fixed":
        required_parameters.add("arrival_interval_hours")

    if service_distribution == "uniform":
        required_parameters.update(
            {
                "minimum_service_time_hours",
                "maximum_service_time_hours",
            }
        )

    for parameter in required_parameters:
        if parameter not in parameters:
            raise ValueError(f"Missing required scenario parameter: {parameter}.")

    duration_hours = parameters["simulation_hours"]

    if not _is_valid_dimension(duration_hours):
        raise ValueError("Simulation hours must be a positive finite number.")

    service_time_hours = parameters["average_service_time_hours"]

    if not _is_valid_dimension(service_time_hours):
        raise ValueError("Average service time must be a positive finite number.")

    expected_arrivals = parameters.get("expected_vessel_arrivals", 0)
    _validate_non_negative_integer(expected_arrivals, "Expected arrivals")

    random_seed = parameters.get("random_seed", DEFAULT_RANDOM_SEED)
    _validate_non_negative_integer(random_seed, "Random seed")

    rng = random.Random(random_seed)

    if service_distribution == "uniform":
        minimum = parameters["minimum_service_time_hours"]
        maximum = parameters["maximum_service_time_hours"]

        if not _is_valid_dimension(minimum) or not _is_valid_dimension(maximum):
            raise ValueError("Service time bounds must be positive finite numbers.")

        if maximum < minimum:
            raise ValueError(
                "Maximum service time must be greater than or equal to minimum."
            )
    elif (
        "minimum_service_time_hours" in parameters
        or "maximum_service_time_hours" in parameters
    ):
        raise ValueError(
            "Minimum and maximum service times can only be supplied "
            "for the uniform distribution."
        )

    vessel_type_priorities = parameters.get("vessel_type_priorities", {})

    if not isinstance(vessel_type_priorities, dict):
        raise ValueError("Vessel type priorities must be a dictionary.")  # noqa: TRY004

    for vessel_type, priority in vessel_type_priorities.items():
        if not isinstance(vessel_type, str) or not vessel_type.strip():
            raise ValueError("Vessel type priority keys must be non-empty strings.")

        _validate_non_negative_integer(priority, "Vessel type priority")

    simulation_started_at = timezone.now()

    if arrival_source == "fixed":
        arrival_interval = parameters["arrival_interval_hours"]

        if not _is_valid_dimension(arrival_interval):
            raise ValueError("Arrival interval must be a positive finite number.")

        vessel_profiles = []

        if expected_arrivals > 0:
            for profile in Vessel.objects.all().order_by("id").iterator():
                if _is_valid_dimension(profile.length_overall) and _is_valid_dimension(
                    profile.draft
                ):
                    vessel_profiles.append(profile)

                    if len(vessel_profiles) >= expected_arrivals:
                        break

            if not vessel_profiles:
                raise ValueError(
                    "At least one vessel profile with valid length and draft "
                    "is required to generate arrivals."
                )

        vessels = [
            _build_simulation_vessel(
                profile=profile,
                vessel_id=f"{profile.id}-arrival-{index + 1}",
                parameters=parameters,
                rng=rng,
            )
            for index in range(expected_arrivals)
            for profile in [vessel_profiles[index % len(vessel_profiles)]]
        ]

    elif arrival_source == "stochastic":
        arrival_interval = None
        vessels = _generate_stochastic_vessels(
            parameters,
            duration_hours,
            rng,
        )
        expected_arrivals = len(vessels)

    else:
        arrival_interval = None
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

            if eta is None or eta.eta_hours >= duration_hours:
                continue

            vessels_with_etas.append((eta.eta_hours, vessel))

        vessels_with_etas.sort(key=lambda item: (item[0], str(item[1].pk)))

        vessels = [
            _build_simulation_vessel(
                profile=vessel,
                vessel_id=vessel.pk,
                parameters=parameters,
                rng=rng,
                arrival_hour=eta_hours,
            )
            for eta_hours, vessel in vessels_with_etas
        ]

        expected_arrivals = len(vessels)

    config = SimulationConfig(
        duration_hours=duration_hours,
        expected_arrivals=expected_arrivals,
        arrival_interval_hours=arrival_interval,
        service_time_hours=service_time_hours,
        vessel_type_priorities=vessel_type_priorities,
    )

    available_berths = parameters.get("available_berths", 2)
    _validate_non_negative_integer(available_berths, "Available berths")

    berth_records = list(
        Berth.objects.filter(status=Berth.OperationalStatus.ACTIVE).order_by("id")[
            :available_berths
        ]
    )

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
