from django.db import transaction
from django.utils import timezone

from port_twin_app.models import SimulationResult, SimulationRun
from port_twin_app.services.simulation_runner import run_scenario_simulation


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

    try:
        output = run_scenario_simulation(simulation_run.scenario)

        with transaction.atomic():
            SimulationResult.objects.create(
                simulation_run=simulation_run,
                result_type="summary",
                data={
                    "total_arrivals": output.summary.total_arrivals,
                    "vessels_served": output.summary.vessels_served,
                    "vessels_waiting": output.summary.vessels_waiting,
                    "vessels_in_service": output.summary.vessels_in_service,
                    "average_waiting_hours": output.summary.average_waiting_hours,
                    "average_turnaround_hours": (
                        output.summary.average_turnaround_hours
                    ),
                    "berth_utilization_percent": (
                        output.summary.berth_utilization_percent
                    ),
                },
            )

            SimulationResult.objects.create(
                simulation_run=simulation_run,
                result_type="vessel_results",
                data={
                    "vessels": [
                        {
                            "vessel_id": result.vessel_id,
                            "arrival_hour": result.arrival_hour,
                            "berth_id": result.berth_id,
                            "service_start_hour": result.service_start_hour,
                            "departure_hour": result.departure_hour,
                            "waiting_hours": result.waiting_hours,
                            "status": result.status,
                            "reason": result.reason,
                        }
                        for result in output.vessel_results
                    ]
                },
            )

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

        return simulation_run

    except Exception as exc:
        with transaction.atomic():
            simulation_run.status = SimulationRun.Status.FAILED
            simulation_run.completed_at = timezone.now()
            simulation_run.error_message = str(exc)
            simulation_run.save(
                update_fields=[
                    "status",
                    "completed_at",
                    "error_message",
                ]
            )
        raise
