"""Django integration helpers for marine AI inference."""

from functools import lru_cache

from port_twin_app.services.marine_ai.inference import (
    MarineInferenceService,
)


@lru_cache(maxsize=2)
def get_marine_inference_service(checkpoint_path):
    """Load and reuse the inference service for a checkpoint."""
    return MarineInferenceService(checkpoint_path=checkpoint_path)
