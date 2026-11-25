import math
from dataclasses import dataclass
from decimal import Decimal
from numbers import Real


@dataclass(frozen=True)
class SimulationVessel:
    """Minimal vessel data required by the simulation engine."""

    vessel_id: str
    length: float
    draft: float
    arrival_hour: float | None = None
    vessel_type: str = "unknown"
    service_time_hours: float | None = None


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
    arrival_interval_hours: float | None
    service_time_hours: float
    vessel_type_priorities: dict[str, int] | None = None
    available_cranes: int = 4
    max_cranes_per_vessel: int = 2


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
    service_time_hours: float | None = None
    service_order: int | None = None
    allocated_cranes: int = 0
    allocated_crane_ids: tuple[int, ...] = ()


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
    baseline_output: "SimulationOutput | None" = None


def _is_finite_number(value: object) -> bool:
    """Return whether a value is a finite real number, excluding booleans."""

    return (
        isinstance(value, (Real, Decimal))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _validate_identifier(value: object, entity: str) -> None:
    """Validate a vessel or berth identifier."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{entity} identifiers must be non-empty strings.")


def run_crane_aware_simulation(
    vessels: list[SimulationVessel],
    berths: list[SimulationBerth],
    config: SimulationConfig,
    policy: str = "optimised",
) -> SimulationOutput:
    """Simulate vessel allocation with limited, shared crane capacity."""

    if policy not in {"baseline", "optimised"}:
        raise ValueError("Policy must be 'baseline' or 'optimised'.")

    # Reuse the existing engine's input validation.
    run_simulation(vessels, berths, config)

    for name, value in (
        ("Available cranes", config.available_cranes),
        ("Maximum cranes per vessel", config.max_cranes_per_vessel),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")

    if config.max_cranes_per_vessel > config.available_cranes:
        raise ValueError("Maximum cranes per vessel cannot exceed available cranes.")

    berth_available_at = {berth.berth_id: 0.0 for berth in berths}
    crane_available_at = {crane_id: 0.0 for crane_id in range(config.available_cranes)}

    arrivals = []
    for index, vessel in enumerate(vessels[: config.expected_arrivals]):
        arrival_hour = (
            vessel.arrival_hour
            if vessel.arrival_hour is not None
            else index * config.arrival_interval_hours
        )
        if arrival_hour < config.duration_hours:
            arrivals.append((arrival_hour, index, vessel))

    arrivals.sort(key=lambda item: (item[0], item[1]))

    results = {}
    waiting = []
    arrival_position = 0
    service_order = 0

    total_waiting = 0.0
    waiting_count = 0
    total_turnaround = 0.0
    completed_count = 0
    occupied_hours = 0.0
    current_time = 0.0

    def compatible(vessel, berth):
        return (
            vessel.length <= berth.max_vessel_length
            and vessel.draft <= berth.max_vessel_draft
        )

    def incompatibility_reason(vessel):
        if not berths:
            return "No berths are configured."

        length_fits = any(vessel.length <= berth.max_vessel_length for berth in berths)
        draft_fits = any(vessel.draft <= berth.max_vessel_draft for berth in berths)

        if not length_fits:
            return "Vessel length exceeds all berth limits."
        if not draft_fits:
            return "Vessel draft exceeds all berth limits."
        return "No single berth can accommodate both the vessel's length and draft."

    priorities = config.vessel_type_priorities or {}

    while arrival_position < len(arrivals) or waiting:
        if current_time >= config.duration_hours:
            break

        # Add every vessel that has arrived by the current event time.
        while (
            arrival_position < len(arrivals)
            and arrivals[arrival_position][0] <= current_time
        ):
            arrival_hour, index, vessel = arrivals[arrival_position]
            arrival_position += 1

            if not any(compatible(vessel, berth) for berth in berths):
                results[index] = VesselSimulationResult(
                    vessel_id=vessel.vessel_id,
                    arrival_hour=arrival_hour,
                    berth_id=None,
                    service_start_hour=None,
                    departure_hour=None,
                    waiting_hours=None,
                    status="incompatible_berth",
                    reason=incompatibility_reason(vessel),
                )
            else:
                waiting.append((arrival_hour, index, vessel))

        # Dispatch to berths that are free at the current event time.
        free_berths = sorted(
            (
                berth
                for berth in berths
                if berth_available_at[berth.berth_id] <= current_time
            ),
            key=lambda berth: berth.berth_id,
        )

        for berth in free_berths:
            candidates = [item for item in waiting if compatible(item[2], berth)]

            if not candidates:
                continue

            arrival_hour, index, vessel = min(
                candidates,
                key=lambda item: (
                    priorities.get(item[2].vessel_type, 0),
                    item[0],
                    item[1],
                ),
            )

            one_crane_service_time = (
                vessel.service_time_hours
                if vessel.service_time_hours is not None
                else config.service_time_hours
            )

            crane_counts = (
                [1]
                if policy == "baseline"
                else range(1, config.max_cranes_per_vessel + 1)
            )

            allocation_candidates = []

            for crane_count in crane_counts:
                available_cranes = sorted(
                    crane_available_at.items(),
                    key=lambda item: (item[1], item[0]),
                )[:crane_count]

                crane_start = available_cranes[-1][1]
                service_start = max(
                    current_time,
                    arrival_hour,
                    berth_available_at[berth.berth_id],
                    crane_start,
                )
                service_time = one_crane_service_time / crane_count
                departure = service_start + service_time

                if service_start >= config.duration_hours:
                    continue

                if policy == "baseline":
                    selection_key = (
                        service_start,
                        departure,
                        crane_count,
                    )
                else:
                    selection_key = (
                        departure,
                        service_start,
                        crane_count,
                    )

                allocation_candidates.append(
                    (
                        selection_key,
                        crane_count,
                        available_cranes,
                        service_start,
                        service_time,
                        departure,
                    )
                )

            if not allocation_candidates:
                continue

            (
                _,
                crane_count,
                allocated_cranes,
                service_start,
                service_time,
                departure,
            ) = min(
                allocation_candidates,
                key=lambda candidate: candidate[0],
            )

            waiting.remove((arrival_hour, index, vessel))

            berth_available_at[berth.berth_id] = departure
            for crane_id, _ in allocated_cranes:
                crane_available_at[crane_id] = departure

            vessel_waiting = service_start - arrival_hour

            if service_start >= config.duration_hours:
                vessel_waiting = config.duration_hours - arrival_hour

            occupied_hours += max(
                0.0,
                min(departure, config.duration_hours) - max(service_start, 0.0),
            )

            if service_start < config.duration_hours:
                total_waiting += vessel_waiting
                waiting_count += 1

            if departure <= config.duration_hours:
                status = "completed"
                total_turnaround += departure - arrival_hour
                completed_count += 1
            elif service_start < config.duration_hours:
                status = "in_service"
            else:
                status = "waiting"

            service_order += 1
            results[index] = VesselSimulationResult(
                vessel_id=vessel.vessel_id,
                arrival_hour=arrival_hour,
                berth_id=berth.berth_id,
                service_start_hour=service_start,
                departure_hour=departure,
                waiting_hours=vessel_waiting,
                status=status,
                service_time_hours=service_time,
                service_order=service_order,
                allocated_cranes=crane_count,
                allocated_crane_ids=tuple(crane_id for crane_id, _ in allocated_cranes),
            )

        # Advance to the next arrival or resource completion.
        next_events = []

        if arrival_position < len(arrivals):
            next_events.append(arrivals[arrival_position][0])

        if waiting:
            for berth in berths:
                if berth_available_at[berth.berth_id] > current_time and any(
                    compatible(item[2], berth) for item in waiting
                ):
                    next_events.append(berth_available_at[berth.berth_id])

            if any(
                crane_available_at[crane_id] > current_time
                for crane_id in crane_available_at
            ):
                next_events.extend(
                    available_at
                    for available_at in crane_available_at.values()
                    if available_at > current_time
                )

        if not next_events:
            break

        current_time = min(next_events)

        if current_time >= config.duration_hours:
            break

    # Process arrivals that were not reached before the simulation horizon.
    for arrival_hour, index, vessel in arrivals[arrival_position:]:
        if not any(compatible(vessel, berth) for berth in berths):
            results[index] = VesselSimulationResult(
                vessel_id=vessel.vessel_id,
                arrival_hour=arrival_hour,
                berth_id=None,
                service_start_hour=None,
                departure_hour=None,
                waiting_hours=None,
                status="incompatible_berth",
                reason=incompatibility_reason(vessel),
            )
        else:
            results[index] = VesselSimulationResult(
                vessel_id=vessel.vessel_id,
                arrival_hour=arrival_hour,
                berth_id=None,
                service_start_hour=None,
                departure_hour=None,
                waiting_hours=config.duration_hours - arrival_hour,
                status="waiting",
            )

    # Record vessels that never started service before the horizon.
    for arrival_hour, index, vessel in waiting:
        results[index] = VesselSimulationResult(
            vessel_id=vessel.vessel_id,
            arrival_hour=arrival_hour,
            berth_id=None,
            service_start_hour=None,
            departure_hour=None,
            waiting_hours=config.duration_hours - arrival_hour,
            status="waiting",
        )

    ordered_results = tuple(results[index] for index in sorted(results))
    total_berth_hours = config.duration_hours * len(berths)

    summary = SimulationSummary(
        total_arrivals=len(ordered_results),
        vessels_served=completed_count,
        vessels_waiting=sum(result.status == "waiting" for result in ordered_results),
        vessels_in_service=sum(
            result.status == "in_service" for result in ordered_results
        ),
        average_waiting_hours=(total_waiting / waiting_count if waiting_count else 0.0),
        average_turnaround_hours=(
            total_turnaround / completed_count if completed_count else 0.0
        ),
        berth_utilization_percent=(
            occupied_hours / total_berth_hours * 100 if total_berth_hours else 0.0
        ),
    )

    return SimulationOutput(
        vessel_results=ordered_results,
        summary=summary,
    )


def _run_priority_simulation(
    vessels: list[SimulationVessel],
    berths: list[SimulationBerth],
    config: SimulationConfig,
) -> SimulationOutput:
    """Run the simulation using vessel-type priorities."""

    priorities = config.vessel_type_priorities or {}
    arrivals = []

    for index, vessel in enumerate(vessels[: config.expected_arrivals]):
        arrival_hour = (
            vessel.arrival_hour
            if vessel.arrival_hour is not None
            else index * config.arrival_interval_hours
        )

        if arrival_hour < config.duration_hours:
            arrivals.append((arrival_hour, index, vessel))

    arrivals.sort(key=lambda item: (item[0], item[1]))

    results: dict[int, VesselSimulationResult] = {}
    waiting: list[tuple[float, int, SimulationVessel]] = []
    berth_available_at = {berth.berth_id: 0.0 for berth in berths}
    service_order = 0

    def compatible(vessel: SimulationVessel, berth: SimulationBerth) -> bool:
        return (
            vessel.length <= berth.max_vessel_length
            and vessel.draft <= berth.max_vessel_draft
        )

    def incompatibility_reason(vessel: SimulationVessel) -> str:
        if not berths:
            return "No berths are configured."

        length_fits = any(vessel.length <= berth.max_vessel_length for berth in berths)
        draft_fits = any(vessel.draft <= berth.max_vessel_draft for berth in berths)

        if not length_fits:
            return "Vessel length exceeds all berth limits."
        if not draft_fits:
            return "Vessel draft exceeds all berth limits."
        return "No single berth can accommodate both the vessel's length and draft."

    total_waiting = 0.0
    waiting_count = 0
    total_turnaround = 0.0
    completed_count = 0
    occupied_hours = 0.0

    current_time = 0.0
    arrival_position = 0

    while arrival_position < len(arrivals) or waiting:
        # Add all vessels that have arrived by the current event time.
        while (
            arrival_position < len(arrivals)
            and arrivals[arrival_position][0] <= current_time
        ):
            arrival_hour, index, vessel = arrivals[arrival_position]
            arrival_position += 1

            if not any(compatible(vessel, berth) for berth in berths):
                results[index] = VesselSimulationResult(
                    vessel_id=vessel.vessel_id,
                    arrival_hour=arrival_hour,
                    berth_id=None,
                    service_start_hour=None,
                    departure_hour=None,
                    waiting_hours=None,
                    status="incompatible_berth",
                    reason=incompatibility_reason(vessel),
                )
            else:
                waiting.append((arrival_hour, index, vessel))

        # Assign free berths to the highest-priority compatible vessels.
        free_berths = sorted(
            (
                berth
                for berth in berths
                if berth_available_at[berth.berth_id] <= current_time
            ),
            key=lambda berth: berth.berth_id,
        )

        for berth in free_berths:
            candidates = [item for item in waiting if compatible(item[2], berth)]

            if not candidates:
                continue

            arrival_hour, index, vessel = min(
                candidates,
                key=lambda item: (
                    priorities.get(item[2].vessel_type, 0),
                    item[0],
                    item[1],
                ),
            )
            waiting.remove((arrival_hour, index, vessel))

            service_time = (
                vessel.service_time_hours
                if vessel.service_time_hours is not None
                else config.service_time_hours
            )
            service_start = max(current_time, arrival_hour)
            departure = service_start + service_time
            vessel_waiting = service_start - arrival_hour

            service_order += 1
            berth_available_at[berth.berth_id] = departure

            occupied_hours += max(
                0.0,
                min(departure, config.duration_hours) - max(service_start, 0.0),
            )

            if service_start < config.duration_hours:
                total_waiting += vessel_waiting
                waiting_count += 1

            if departure <= config.duration_hours:
                status = "completed"
                total_turnaround += departure - arrival_hour
                completed_count += 1
            elif service_start < config.duration_hours:
                status = "in_service"
            else:
                status = "waiting"
                vessel_waiting = config.duration_hours - arrival_hour

            results[index] = VesselSimulationResult(
                vessel_id=vessel.vessel_id,
                arrival_hour=arrival_hour,
                berth_id=berth.berth_id,
                service_start_hour=service_start,
                departure_hour=departure,
                waiting_hours=vessel_waiting,
                status=status,
                service_time_hours=service_time,
                service_order=service_order,
            )

        # Advance to the next arrival or relevant berth completion.
        next_events = []

        if arrival_position < len(arrivals):
            next_events.append(arrivals[arrival_position][0])

        if waiting:
            for berth in berths:
                if berth_available_at[berth.berth_id] > current_time and any(
                    compatible(item[2], berth) for item in waiting
                ):
                    next_events.append(berth_available_at[berth.berth_id])

        if not next_events:
            break

        current_time = min(next_events)

    # Vessels still waiting when the horizon ends have not started service.
    for arrival_hour, index, vessel in waiting:
        results[index] = VesselSimulationResult(
            vessel_id=vessel.vessel_id,
            arrival_hour=arrival_hour,
            berth_id=None,
            service_start_hour=None,
            departure_hour=None,
            waiting_hours=config.duration_hours - arrival_hour,
            status="waiting",
        )

    ordered_results = tuple(results[index] for index in sorted(results))
    total_berth_hours = config.duration_hours * len(berths)

    summary = SimulationSummary(
        total_arrivals=len(ordered_results),
        vessels_served=completed_count,
        vessels_waiting=sum(result.status == "waiting" for result in ordered_results),
        vessels_in_service=sum(
            result.status == "in_service" for result in ordered_results
        ),
        average_waiting_hours=(total_waiting / waiting_count if waiting_count else 0.0),
        average_turnaround_hours=(
            total_turnaround / completed_count if completed_count else 0.0
        ),
        berth_utilization_percent=(
            occupied_hours / total_berth_hours * 100 if total_berth_hours else 0.0
        ),
    )

    return SimulationOutput(
        vessel_results=ordered_results,
        summary=summary,
    )


def run_simulation(
    vessels: list[SimulationVessel],
    berths: list[SimulationBerth],
    config: SimulationConfig,
) -> SimulationOutput:
    """Simulate vessel arrivals and allocation to compatible berths."""

    # Validate simulation configuration.
    positive_config_values = {
        "Simulation duration": config.duration_hours,
        "Service time": config.service_time_hours,
    }

    for name, value in positive_config_values.items():
        if not _is_finite_number(value) or value <= 0:
            raise ValueError(f"{name} must be a finite number greater than zero.")

    interval = config.arrival_interval_hours

    if interval is not None and (not _is_finite_number(interval) or interval < 0):
        raise ValueError(
            "Arrival interval must be a finite number greater than or equal to zero."
        )

    if interval is None and any(vessel.arrival_hour is None for vessel in vessels):
        raise ValueError(
            "Every vessel must have an explicit arrival hour when no arrival "
            "interval is configured."
        )

    for vessel in vessels:
        if vessel.arrival_hour is not None and (
            not _is_finite_number(vessel.arrival_hour) or vessel.arrival_hour < 0
        ):
            raise ValueError(
                "Vessel arrival hours must be finite numbers greater than "
                "or equal to zero."
            )

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

    for vessel_id in vessel_ids:
        _validate_identifier(vessel_id, "Vessel")

    if len(vessel_ids) != len(set(vessel_ids)):
        raise ValueError("Vessel identifiers must be unique.")

    for vessel in vessels:
        if (
            not _is_finite_number(vessel.length)
            or not _is_finite_number(vessel.draft)
            or vessel.length <= 0
            or vessel.draft <= 0
        ):
            raise ValueError(
                "Vessel length and draft must be finite numbers greater than zero."
            )

        for vessel in vessels:
            if (
                not isinstance(vessel.vessel_type, str)
                or not vessel.vessel_type.strip()
            ):
                raise ValueError("Vessel types must be non-empty strings.")

            if vessel.service_time_hours is not None and (
                not _is_finite_number(vessel.service_time_hours)
                or vessel.service_time_hours <= 0
            ):
                raise ValueError(
                    "Vessel service times must be finite numbers greater than zero."
                )

        priorities = config.vessel_type_priorities

        if priorities is not None:
            if not isinstance(priorities, dict):
                raise ValueError("Vessel type priorities must be a dictionary.")

            for vessel_type, priority in priorities.items():
                if not isinstance(vessel_type, str) or not vessel_type.strip():
                    raise ValueError("Priority vessel types must be non-empty strings.")

                if (
                    isinstance(priority, bool)
                    or not isinstance(priority, int)
                    or priority < 0
                ):
                    raise ValueError(
                        "Vessel type priorities must be non-negative integers."
                    )

    # Validate berth capacity limits and identifiers.
    berth_ids = [berth.berth_id for berth in berths]

    for berth_id in berth_ids:
        _validate_identifier(berth_id, "Berth")

    if len(berth_ids) != len(set(berth_ids)):
        raise ValueError("Berth identifiers must be unique.")

    for berth in berths:
        if (
            not _is_finite_number(berth.max_vessel_length)
            or not _is_finite_number(berth.max_vessel_draft)
            or berth.max_vessel_length <= 0
            or berth.max_vessel_draft <= 0
        ):
            raise ValueError(
                "Berth capacity limits must be finite numbers greater than zero."
            )

    if config.vessel_type_priorities:
        return _run_priority_simulation(vessels, berths, config)

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
        arrival_hour = (
            vessel.arrival_hour
            if vessel.arrival_hour is not None
            else index * config.arrival_interval_hours
        )

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
                    vessel.length <= berth.max_vessel_length for berth in berths
                )
                draft_fits_some_berth = any(
                    vessel.draft <= berth.max_vessel_draft for berth in berths
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
        service_time = (
            vessel.service_time_hours
            if vessel.service_time_hours is not None
            else config.service_time_hours
        )

        if not _is_finite_number(service_time) or service_time <= 0:
            raise ValueError(
                "Vessel service times must be finite numbers greater than zero."
            )

        departure = service_start + service_time

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
                service_time_hours=service_time,
                service_order=index + 1,
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
