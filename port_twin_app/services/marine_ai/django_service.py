"""Django integration helpers for marine AI inference."""

from functools import lru_cache
from pathlib import Path

from port_twin_app.services.marine_ai.inference import (
    MarineInferenceService,
)


@lru_cache(maxsize=2)
def get_marine_inference_service(checkpoint_path):
    """Cache inference services by their configured checkpoint path."""
    return MarineInferenceService(checkpoint_path=Path(checkpoint_path))
