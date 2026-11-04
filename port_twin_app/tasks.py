import logging

from celery import shared_task

from port_twin_app.models import SimulationRun
from port_twin_app.services.simulation_execution import execute_simulation_run

logger = logging.getLogger(__name__)


@shared_task(name="port_twin_app.execute_simulation")
def execute_simulation_task(simulation_run_id):
    """Execute a simulation run in a Celery worker."""

    try:
        simulation_run = SimulationRun.objects.get(pk=simulation_run_id)
    except SimulationRun.DoesNotExist:
        logger.error(
            "Simulation run not found: run_id=%s",
            simulation_run_id,
        )
        return

    try:
        execute_simulation_run(simulation_run)
    except ValueError:
        # A run that is no longer pending must not be executed again.
        logger.warning(
            "Simulation execution skipped: run_id=%s status=%s",
            simulation_run_id,
            simulation_run.status,
        )
    except Exception:
        # The execution service already records the failed status.
        # Celery logs the task failure without exposing details to the API.
        logger.exception(
            "Simulation task failed: run_id=%s",
            simulation_run_id,
        )
        raise
