import logging

from django.db import transaction
from django.utils import timezone

from port_twin_app.models import MarinePrediction, SimulationResult, SimulationRun
from port_twin_app.services.marine_ai.conditions import (
    build_marine_condition_advisory,
)
from port_twin_app.services.simulation_runner import run_scenario_simulation

logger = logging.getLogger(__name__)


def _serialize_summary(summary):
    """Convert simulation summary metrics into JSON-compatible data."""

    return {
        "total_arrivals": summary.total_arrivals,
        "vessels_served": summary.vessels_served,
        "vessels_waiting": summary.vessels_waiting,
        "vessels_in_service": summary.vessels_in_service,
        "average_waiting_hours": summary.average_waiting_hours,
        "average_turnaround_hours": summary.average_turnaround_hours,
        "berth_utilization_percent": summary.berth_utilization_percent,
    }


def _serialize_vessel_results(vessel_results):
    """Convert individual vessel outcomes into JSON-compatible data."""

    return {
        "vessels": [
            {
                "vessel_id": result.vessel_id,
                "arrival_hour": result.arrival_hour,
                "berth_id": result.berth_id,
                "service_start_hour": result.service_start_hour,
                "service_time_hours": result.service_time_hours,
                "service_order": result.service_order,
                "departure_hour": result.departure_hour,
                "waiting_hours": result.waiting_hours,
                "status": result.status,
                "reason": result.reason,
                "allocated_cranes": result.allocated_cranes,
                "allocated_crane_ids": list(result.allocated_crane_ids),
            }
            for result in vessel_results
        ]
    }


def _build_crane_schedule(vessel_results):
    """Build a schedule containing each crane's assigned service period."""

    assignments = []

    for result in vessel_results:
        if (
            result.service_start_hour is None
            or result.departure_hour is None
            or not result.allocated_crane_ids
        ):
            continue

        for crane_id in result.allocated_crane_ids:
            assignments.append(
                {
                    "crane_id": crane_id,
                    "vessel_id": result.vessel_id,
                    "berth_id": result.berth_id,
                    "start_hour": result.service_start_hour,
                    "end_hour": result.departure_hour,
                }
            )

    assignments.sort(
        key=lambda assignment: (
            assignment["start_hour"],
            assignment["crane_id"],
        )
    )

    return {
        "unit": "simulation_hours",
        "assignments": assignments,
    }


def _build_policy_comparison(baseline_summary, optimised_summary):
    """Compare baseline and optimised simulation metrics."""

    baseline = _serialize_summary(baseline_summary)
    optimised = _serialize_summary(optimised_summary)

    comparable_metrics = (
        "total_arrivals",
        "vessels_served",
        "vessels_waiting",
        "vessels_in_service",
        "average_waiting_hours",
        "average_turnaround_hours",
        "berth_utilization_percent",
    )

    return {
        "baseline": baseline,
        "optimised": optimised,
        "optimised_minus_baseline": {
            metric: optimised[metric] - baseline[metric]
            for metric in comparable_metrics
        },
    }


def execute_simulation_run(simulation_run: SimulationRun):
    """Execute a pending simulation run and persist its results."""

    with transaction.atomic():
        locked_run = SimulationRun.objects.select_for_update().get(pk=simulation_run.pk)

        if locked_run.status != SimulationRun.Status.PENDING:
            raise ValueError("Only pending simulation runs can be executed.")

        locked_run.status = SimulationRun.Status.RUNNING
        locked_run.started_at = timezone.now()
        locked_run.completed_at = None
        locked_run.error_message = ""
        locked_run.save(
            update_fields=[
                "status",
                "started_at",
                "completed_at",
                "error_message",
            ]
        )

    simulation_run.refresh_from_db()

    logger.info(
        "Simulation started: run_id=%s scenario_id=%s",
        simulation_run.pk,
        simulation_run.scenario_id,
    )

    try:
        output = run_scenario_simulation(simulation_run.scenario)
        marine_conditions = None
        prediction_id = simulation_run.scenario.parameters.get("marine_prediction_id")
        nominal_speed_schedule = simulation_run.scenario.parameters.get(
            "nominal_piloting_speed_schedule"
        )

        if prediction_id is not None:
            prediction = MarinePrediction.objects.get(
                pk=prediction_id,
                status=MarinePrediction.Status.COMPLETED,
            )
            marine_conditions = build_marine_condition_advisory(prediction)

        with transaction.atomic():
            SimulationResult.objects.create(
                simulation_run=simulation_run,
                result_type="summary",
                data=_serialize_summary(output.summary),
            )

            SimulationResult.objects.create(
                simulation_run=simulation_run,
                result_type="vessel_results",
                data=_serialize_vessel_results(output.vessel_results),
            )

            # Persist baseline and optimisation comparison when available.
            if output.baseline_output is not None:
                baseline_output = output.baseline_output

                SimulationResult.objects.create(
                    simulation_run=simulation_run,
                    result_type="baseline_summary",
                    data=_serialize_summary(baseline_output.summary),
                )

                SimulationResult.objects.create(
                    simulation_run=simulation_run,
                    result_type="baseline_vessel_results",
                    data=_serialize_vessel_results(baseline_output.vessel_results),
                )

                SimulationResult.objects.create(
                    simulation_run=simulation_run,
                    result_type="crane_schedule",
                    data=_build_crane_schedule(output.vessel_results),
                )

                SimulationResult.objects.create(
                    simulation_run=simulation_run,
                    result_type="policy_comparison",
                    data=_build_policy_comparison(
                        baseline_output.summary,
                        output.summary,
                    ),
                )

            # Save marine advisory if a prediction was pinned to the scenario.
            if marine_conditions is not None:
                SimulationResult.objects.create(
                    simulation_run=simulation_run,
                    result_type="marine_conditions",
                    data=marine_conditions,
                )

            # Save configured nominal speeds as planning assumptions.
            if nominal_speed_schedule is not None:
                SimulationResult.objects.create(
                    simulation_run=simulation_run,
                    result_type="nominal_piloting_speed_schedule",
                    data={
                        "scope": "all_vessels",
                        "unit": "knots",
                        "label": "Nominal piloting speed assumptions",
                        "description": (
                            "Configured planning assumptions only. "
                            "These are not navigational advice or "
                            "current-adjusted speed recommendations."
                        ),
                        "periods": nominal_speed_schedule,
                    },
                )

            # Mark the simulation as completed only after all results are saved.
            simulation_run.status = SimulationRun.Status.COMPLETED
            simulation_run.completed_at = timezone.now()
            simulation_run.error_message = ""
            simulation_run.save(
                update_fields=[
                    "status",
                    "completed_at",
                    "error_message",
                ]
            )

        logger.info(
            "Simulation completed: run_id=%s scenario_id=%s",
            simulation_run.pk,
            simulation_run.scenario_id,
        )
        return simulation_run

    except Exception as exc:
        with transaction.atomic():
            simulation_run.status = SimulationRun.Status.FAILED
            simulation_run.completed_at = timezone.now()
            simulation_run.error_message = (
                "Simulation execution failed. Check server logs for details."
            )
            simulation_run.save(
                update_fields=[
                    "status",
                    "completed_at",
                    "error_message",
                ]
            )

        # Log identifiers and the exception class, not its potentially sensitive
        # message or traceback.
        logger.error(
            "Simulation failed: run_id=%s scenario_id=%s exception_type=%s",
            simulation_run.pk,
            simulation_run.scenario_id,
            type(exc).__name__,
        )
        raise
