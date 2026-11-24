import logging

from celery import shared_task

from port_twin_app.models import SimulationRun
from port_twin_app.services.marine_ai.prediction_execution import (
    execute_marine_prediction,
)
from port_twin_app.services.marine_data.transit.execution import (
    execute_transit_assessment,
)
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
    except Exception as exc:
        logger.error(
            "Simulation task failed: run_id=%s exception_type=%s",
            simulation_run_id,
            type(exc).__name__,
        )
        raise


@shared_task(name="port_twin_app.execute_marine_prediction")
def execute_marine_prediction_task(prediction_id):
    """Execute a marine prediction in a Celery worker."""

    try:
        execute_marine_prediction(prediction_id)
    except Exception as exc:
        logger.error(
            "Marine prediction task failed: prediction_id=%s exception_type=%s",
            prediction_id,
            type(exc).__name__,
        )
        raise


@shared_task(name="port_twin_app.execute_transit_assessment")
def execute_transit_assessment_task(assessment_id):
    """Execute a transit assessment in a Celery worker."""

    try:
        execute_transit_assessment(assessment_id)
    except Exception as exc:
        logger.error(
            "Transit assessment task failed: assessment_id=%s exception_type=%s",
            assessment_id,
            type(exc).__name__,
        )
        raise
