"""Django integration helpers for marine AI inference."""

from functools import lru_cache
from pathlib import Path

from django.conf import settings

from port_twin_app.services.marine_ai.inference import (
    MarineInferenceService,
)
from port_twin_app.services.marine_ai.modelarts import (
    ModelArtsInferenceService,
)


@lru_cache(maxsize=2)
def get_marine_inference_service(
    checkpoint_path,
    backend="local",
    endpoint_url="",
    token="",
    timeout=30.0,
    model_version="modelarts-unknown",
):
    """Return the configured local or remote inference service."""
    if backend == "local":
        return MarineInferenceService(
            checkpoint_path=Path(checkpoint_path),
        )

    if backend == "modelarts":
        if not endpoint_url:
            raise ValueError(
                "MARINE_AI_MODELARTS_ENDPOINT must be configured "
                "when MARINE_AI_BACKEND=modelarts."
            )

        return ModelArtsInferenceService(
            endpoint_url=endpoint_url,
            token=token,
            timeout=timeout,
            model_version=model_version,
        )

    raise ValueError(f"Unsupported marine AI backend: {backend}")


def get_configured_marine_inference_service():
    """Resolve the service from Django settings."""
    return get_marine_inference_service(
        checkpoint_path=settings.MARINE_AI_CHECKPOINT,
        backend=settings.MARINE_AI_BACKEND,
        endpoint_url=settings.MARINE_AI_MODELARTS_ENDPOINT,
        token=settings.MARINE_AI_MODELARTS_TOKEN,
        timeout=settings.MARINE_AI_MODELARTS_TIMEOUT,
        model_version=settings.MARINE_AI_MODELARTS_VERSION,
    )
