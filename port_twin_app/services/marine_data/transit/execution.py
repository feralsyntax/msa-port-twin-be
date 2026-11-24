import logging

from django.db import transaction
from django.utils import timezone

from port_twin_app.models import TransitAssessment
from port_twin_app.services.marine_data.transit.assessment import (
    run_transit_assessment,
)
from port_twin_app.services.marine_data.transit.config import (
    DraftClass,
    TransitAssessmentConfig,
)

logger = logging.getLogger(__name__)

SAFE_ERROR_MESSAGE = "Transit assessment failed. Check server logs for details."


def _build_assessment_config(configuration):
    """Reconstruct the assessment configuration from its stored snapshot."""

    configuration = dict(configuration or {})

    draft_classes = configuration.pop("draft_classes", None)

    if draft_classes is not None:
        configuration["draft_classes"] = tuple(
            DraftClass(**draft_class) for draft_class in draft_classes
        )

    return TransitAssessmentConfig(**configuration)


def execute_transit_assessment(assessment_id):
    """Execute a pending transit assessment and persist its outcome."""

    with transaction.atomic():
        assessment = TransitAssessment.objects.select_for_update().get(pk=assessment_id)

        if assessment.status != TransitAssessment.Status.PENDING:
            return assessment

        assessment.status = TransitAssessment.Status.RUNNING
        assessment.started_at = timezone.now()
        assessment.completed_at = None
        assessment.error_message = ""
        assessment.save(
            update_fields=[
                "status",
                "started_at",
                "completed_at",
                "error_message",
                "updated_at",
            ]
        )

    try:
        configuration = _build_assessment_config(assessment.configuration)

        result = run_transit_assessment(
            scenario_name=assessment.scenario_name,
            domain_name=assessment.domain_name,
            assessment_config=configuration,
        )

        with transaction.atomic():
            assessment = TransitAssessment.objects.select_for_update().get(
                pk=assessment_id
            )

            assessment.result_data = result
            assessment.status = TransitAssessment.Status.COMPLETED
            assessment.completed_at = timezone.now()
            assessment.error_message = ""
            assessment.save(
                update_fields=[
                    "result_data",
                    "status",
                    "completed_at",
                    "error_message",
                    "updated_at",
                ]
            )

        logger.info(
            "Transit assessment completed: assessment_id=%s",
            assessment_id,
        )

        return assessment

    except Exception as exc:
        logger.error(
            "Transit assessment failed: assessment_id=%s exception_type=%s",
            assessment_id,
            type(exc).__name__,
        )

        with transaction.atomic():
            assessment = TransitAssessment.objects.select_for_update().get(
                pk=assessment_id
            )

            assessment.status = TransitAssessment.Status.FAILED
            assessment.completed_at = timezone.now()
            assessment.error_message = SAFE_ERROR_MESSAGE
            assessment.save(
                update_fields=[
                    "status",
                    "completed_at",
                    "error_message",
                    "updated_at",
                ]
            )

        raise
