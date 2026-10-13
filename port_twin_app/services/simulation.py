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

    if config.duration_hours <= 0:
        raise ValueError("Simulation duration must be greater than zero.")

    if config.expected_arrivals < 0:
        raise ValueError("Expected arrivals cannot be negative.")

    if config.arrival_interval_hours <= 0:
        raise ValueError("Arrival interval must be greater than zero.")

    if config.service_time_hours <= 0:
        raise ValueError("Service time must be greater than zero.")

    if config.expected_arrivals > len(vessels):
        raise ValueError(
            "The number of expected arrivals cannot exceed the supplied vessels."
        )

    if any(v.length <= 0 or v.draft <= 0 for v in vessels):
        raise ValueError("Vessel length and draft must be greater than zero.")

    if any(b.max_vessel_length <= 0 or b.max_vessel_draft <= 0 for b in berths):
        raise ValueError("Berth capacity limits must be greater than zero.")

    # Each berth tracks the hour when it next becomes available.
    berth_available_at = {berth.berth_id: 0.0 for berth in berths}
    results: list[VesselSimulationResult] = []

    total_waiting = 0.0
    waiting_count = 0
    total_turnaround = 0.0
    completed_count = 0
    occupied_hours = 0.0

    for index, vessel in enumerate(vessels[: config.expected_arrivals]):
        arrival_hour = index * config.arrival_interval_hours

        # Only vessels arriving within the simulation window are included.
        if arrival_hour >= config.duration_hours:
            break

        compatible_berths = [
            berth
            for berth in berths
            if vessel.length <= berth.max_vessel_length
            and vessel.draft <= berth.max_vessel_draft
        ]

        if not compatible_berths:
            results.append(
                VesselSimulationResult(
                    vessel_id=vessel.vessel_id,
                    arrival_hour=arrival_hour,
                    berth_id=None,
                    service_start_hour=None,
                    departure_hour=None,
                    waiting_hours=None,
                    status="incompatible_berth",
                )
            )
            continue

        # Pick the compatible berth available soonest.
        berth = min(
            compatible_berths,
            key=lambda item: (
                max(berth_available_at[item.berth_id], arrival_hour),
                item.berth_id,
            ),
        )

        service_start = max(
            berth_available_at[berth.berth_id],
            arrival_hour,
        )
        departure = service_start + config.service_time_hours
        waiting = service_start - arrival_hour

        berth_available_at[berth.berth_id] = departure

        # Count only the portion of service occurring within the horizon.
        occupied_hours += max(
            0.0,
            min(departure, config.duration_hours) - max(service_start, 0.0),
        )

        if waiting > 0:
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
        vessels_waiting=sum(r.status == "waiting" for r in results),
        vessels_in_service=sum(r.status == "in_service" for r in results),
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
