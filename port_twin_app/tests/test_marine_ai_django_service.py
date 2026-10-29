from unittest.mock import patch

from django.test import SimpleTestCase, override_settings

from port_twin_app.services.marine_ai.django_service import (
    get_configured_marine_inference_service,
)


class ConfiguredMarineInferenceServiceTests(SimpleTestCase):
    @override_settings(
        MARINE_AI_BACKEND="modelarts",
        MARINE_AI_CHECKPOINT="/tmp/unused.pt",
        MARINE_AI_MODELARTS_ENDPOINT="https://example.com/inference",
        MARINE_AI_MODELARTS_TOKEN="test-token",
        MARINE_AI_MODELARTS_TIMEOUT=12.0,
        MARINE_AI_MODELARTS_VERSION="modelarts-test",
    )
    def test_modelarts_backend_uses_configured_settings(self):
        get_service = (
            "port_twin_app.services.marine_ai.django_service.ModelArtsInferenceService"
        )

        with patch(get_service) as mock_service:
            result = get_configured_marine_inference_service()

        mock_service.assert_called_once_with(
            endpoint_url="https://example.com/inference",
            token="test-token",
            timeout=12.0,
            model_version="modelarts-test",
        )
        self.assertEqual(result, mock_service.return_value)

    @override_settings(
        MARINE_AI_BACKEND="modelarts",
        MARINE_AI_CHECKPOINT="/tmp/unused.pt",
        MARINE_AI_MODELARTS_ENDPOINT="",
        MARINE_AI_MODELARTS_TOKEN="",
        MARINE_AI_MODELARTS_TIMEOUT=30.0,
        MARINE_AI_MODELARTS_VERSION="modelarts-test",
    )
    def test_modelarts_backend_requires_endpoint(self):
        with self.assertRaisesMessage(
            ValueError,
            "MARINE_AI_MODELARTS_ENDPOINT must be configured",
        ):
            get_configured_marine_inference_service()

    @override_settings(
        MARINE_AI_BACKEND="unsupported",
        MARINE_AI_CHECKPOINT="/tmp/unused.pt",
        MARINE_AI_MODELARTS_ENDPOINT="",
        MARINE_AI_MODELARTS_TOKEN="",
        MARINE_AI_MODELARTS_TIMEOUT=30.0,
        MARINE_AI_MODELARTS_VERSION="modelarts-test",
    )
    def test_unsupported_backend_is_rejected(self):
        with self.assertRaisesMessage(
            ValueError,
            "Unsupported marine AI backend: unsupported",
        ):
            get_configured_marine_inference_service()
