import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationVessel:
    """Minimal vessel data required by the simulation engine."""

    vessel_id: str
    length: float
    draft: float


@dataclass(frozen=True)
class SimulationBerth:
    """A berth and the vessel dimensions it can accommodate."""

    berth_id: str
    max_vessel_length: float
    max_vessel_draft: float


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for one queueing simulation."""

    duration_hours: float
    expected_arrivals: int
    arrival_interval_hours: float
    service_time_hours: float


@dataclass(frozen=True)
class VesselSimulationResult:
    """Simulated outcome for an individual vessel."""

    vessel_id: str
    arrival_hour: float
    berth_id: str | None
    service_start_hour: float | None
    departure_hour: float | None
    waiting_hours: float | None
    status: str
    reason: str | None = None


@dataclass(frozen=True)
class SimulationSummary:
    """Aggregated results for a simulation run."""

    total_arrivals: int
    vessels_served: int
    vessels_waiting: int
    vessels_in_service: int
    average_waiting_hours: float
    average_turnaround_hours: float
    berth_utilization_percent: float


@dataclass(frozen=True)
class SimulationOutput:
    """Complete output from a simulation run."""

    vessel_results: tuple[VesselSimulationResult, ...]
    summary: SimulationSummary


def run_simulation(
    vessels: list[SimulationVessel],
    berths: list[SimulationBerth],
    config: SimulationConfig,
) -> SimulationOutput:
    """Simulate vessel arrivals and allocation to compatible berths."""

    # Validate simulation configuration.
    config_values = {
        "Simulation duration": config.duration_hours,
        "Arrival interval": config.arrival_interval_hours,
        "Service time": config.service_time_hours,
    }

    for name, value in config_values.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite number greater than zero.")

    if isinstance(config.expected_arrivals, bool) or not isinstance(
        config.expected_arrivals, int
    ):
        raise ValueError("Expected arrivals must be an integer.")  # noqa: TRY004

    if config.expected_arrivals < 0:
        raise ValueError("Expected arrivals cannot be negative.")

    if config.expected_arrivals > len(vessels):
        raise ValueError(
            "The number of expected arrivals cannot exceed the supplied vessels."
        )

    # Validate vessel dimensions and identifiers.
    vessel_ids = [vessel.vessel_id for vessel in vessels]

    if len(vessel_ids) != len(set(vessel_ids)):
        raise ValueError("Vessel identifiers must be unique.")

    for vessel in vessels:
        if (
            not math.isfinite(vessel.length)
            or not math.isfinite(vessel.draft)
            or vessel.length <= 0
            or vessel.draft <= 0
        ):
            raise ValueError(
                "Vessel length and draft must be finite numbers greater than zero."
            )

    # Validate berth capacity limits and identifiers.
    berth_ids = [berth.berth_id for berth in berths]

    if len(berth_ids) != len(set(berth_ids)):
        raise ValueError("Berth identifiers must be unique.")

    for berth in berths:
        if (
            not math.isfinite(berth.max_vessel_length)
            or not math.isfinite(berth.max_vessel_draft)
            or berth.max_vessel_length <= 0
            or berth.max_vessel_draft <= 0
        ):
            raise ValueError(
                "Berth capacity limits must be finite numbers greater than zero."
            )

    # Track the next available hour for each berth.
    berth_available_at = {berth.berth_id: 0.0 for berth in berths}

    results: list[VesselSimulationResult] = []

    total_waiting = 0.0
    waiting_count = 0
    total_turnaround = 0.0
    completed_count = 0
    occupied_hours = 0.0

    # Each supplied simulation vessel represents one arrival event.
    for index, vessel in enumerate(vessels[: config.expected_arrivals]):
        arrival_hour = index * config.arrival_interval_hours

        # Only arrivals within the simulation horizon are included.
        if arrival_hour >= config.duration_hours:
            break

        compatible_berths = [
            berth
            for berth in berths
            if vessel.length <= berth.max_vessel_length
            and vessel.draft <= berth.max_vessel_draft
        ]

        # No berth can accommodate this vessel.
        if not compatible_berths:
            if not berths:
                reason = "No berths are configured."
            else:
                reasons = []

                length_fits_some_berth = any(
                    vessel.length <= berth.max_vessel_length
                    for berth in berths
                )
                draft_fits_some_berth = any(
                    vessel.draft <= berth.max_vessel_draft
                    for berth in berths
                )

                if not length_fits_some_berth:
                    reasons.append("Vessel length exceeds all berth limits.")

                if not draft_fits_some_berth:
                    reasons.append("Vessel draft exceeds all berth limits.")

                if length_fits_some_berth and draft_fits_some_berth:
                    reasons.append(
                        "No single berth can accommodate both the vessel's "
                        "length and draft."
                    )

                reason = " ".join(reasons)

            results.append(
                VesselSimulationResult(
                    vessel_id=vessel.vessel_id,
                    arrival_hour=arrival_hour,
                    berth_id=None,
                    service_start_hour=None,
                    departure_hour=None,
                    waiting_hours=None,
                    status="incompatible_berth",
                    reason=reason,
                )
            )
            continue

        # Choose the compatible berth available soonest.
        # Use berth ID as a deterministic tie-breaker.
        berth = min(
            compatible_berths,
            key=lambda item: (
                max(
                    berth_available_at[item.berth_id],
                    arrival_hour,
                ),
                item.berth_id,
            ),
        )

        service_start = max(
            berth_available_at[berth.berth_id],
            arrival_hour,
        )
        departure = service_start + config.service_time_hours
        waiting = service_start - arrival_hour

        if service_start >= config.duration_hours:
            waiting = config.duration_hours - arrival_hour

        berth_available_at[berth.berth_id] = departure

        # Count only service hours occurring within the simulation horizon.
        occupied_hours += max(
            0.0,
            min(departure, config.duration_hours) - max(service_start, 0.0),
        )

        if service_start < config.duration_hours:
            total_waiting += waiting
            waiting_count += 1

        if departure <= config.duration_hours:
            status = "completed"
            total_turnaround += departure - arrival_hour
            completed_count += 1
        elif service_start < config.duration_hours:
            status = "in_service"
        else:
            status = "waiting"

        results.append(
            VesselSimulationResult(
                vessel_id=vessel.vessel_id,
                arrival_hour=arrival_hour,
                berth_id=berth.berth_id,
                service_start_hour=service_start,
                departure_hour=departure,
                waiting_hours=waiting,
                status=status,
            )
        )

    horizon = config.duration_hours
    total_berth_hours = horizon * len(berths)

    summary = SimulationSummary(
        total_arrivals=len(results),
        vessels_served=completed_count,
        vessels_waiting=sum(result.status == "waiting" for result in results),
        vessels_in_service=sum(result.status == "in_service" for result in results),
        average_waiting_hours=(total_waiting / waiting_count if waiting_count else 0.0),
        average_turnaround_hours=(
            total_turnaround / completed_count if completed_count else 0.0
        ),
        berth_utilization_percent=(
            occupied_hours / total_berth_hours * 100 if total_berth_hours else 0.0
        ),
    )

    return SimulationOutput(
        vessel_results=tuple(results),
        summary=summary,
    )
