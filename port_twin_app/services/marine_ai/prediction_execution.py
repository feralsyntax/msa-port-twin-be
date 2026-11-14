import logging

from django.db import transaction

from port_twin_app.models import MarinePrediction
from port_twin_app.services.marine_ai.contracts import (
    MarineForecastConfig,
    MarineGrid,
    MarinePredictionRequest,
)
from port_twin_app.services.marine_ai.django_service import (
    get_configured_marine_inference_service,
)

logger = logging.getLogger(__name__)

SAFE_ERROR_MESSAGE = "Marine prediction failed. Check server logs for details."


def _build_prediction_request(prediction):
    """Reconstruct and validate the stored inference request."""

    validated_inputs = prediction.input_data["validated_inputs"]

    return MarinePredictionRequest(
        grid=MarineGrid(**prediction.grid),
        forecast=MarineForecastConfig(**prediction.forecast),
        **validated_inputs,
    )


def execute_marine_prediction(prediction_id):
    """Execute a pending marine prediction and persist its outcome."""

    with transaction.atomic():
        prediction = MarinePrediction.objects.select_for_update().get(pk=prediction_id)

        if prediction.status != MarinePrediction.Status.PENDING:
            return prediction

        prediction.status = MarinePrediction.Status.RUNNING
        prediction.save(update_fields=["status"])

    try:
        request = _build_prediction_request(prediction)
        service = get_configured_marine_inference_service()
        result = service.predict(request)

        result.validate_against(request)

        result_data = {
            "water_level_m": result.water_level_m,
            "current_u_ms": result.current_u_ms,
            "current_v_ms": result.current_v_ms,
            "status": result.status,
        }

        with transaction.atomic():
            prediction = MarinePrediction.objects.select_for_update().get(
                pk=prediction_id
            )

            prediction.result_data = result_data
            prediction.model_version = result.model_version
            prediction.contract_version = result.contract_version
            prediction.prediction_mode = getattr(service, "prediction_mode", "residual")
            prediction.status = MarinePrediction.Status.COMPLETED
            prediction.error_message = ""
            prediction.save(
                update_fields=[
                    "result_data",
                    "model_version",
                    "contract_version",
                    "prediction_mode",
                    "status",
                    "error_message",
                ]
            )

        return prediction

    except Exception as exc:
        logger.error(
            "Marine prediction failed: prediction_id=%s exception_type=%s",
            prediction_id,
            type(exc).__name__,
        )

        with transaction.atomic():
            prediction = MarinePrediction.objects.select_for_update().get(
                pk=prediction_id
            )

            prediction.status = MarinePrediction.Status.FAILED
            prediction.error_message = SAFE_ERROR_MESSAGE
            prediction.save(update_fields=["status", "error_message"])

        raise
