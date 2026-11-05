"""Integration tests verifying inference with the trained marine FNO artifact."""

import math
from pathlib import Path

from django.test import SimpleTestCase, override_settings
from rest_framework import status
from rest_framework.test import APITestCase

from port_twin_app.models import MarinePrediction
from port_twin_app.services.marine_ai.contracts import (
    CONTRACT_VERSION,
)
from port_twin_app.services.marine_ai.django_service import (
    get_marine_inference_service,
)
from port_twin_app.services.marine_ai.inference import (
    MarineInferenceService,
)
from port_twin_app.tests.test_marine_ai_inference import make_request
from port_twin_app.tests.test_marine_prediction_api import make_api_payload

CHECKPOINT_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "marine_fno.pt"


class TrainedMarineFNOInferenceTests(SimpleTestCase):
    """Verify inference using the committed trained checkpoint."""

    def test_trained_checkpoint_produces_valid_marine_forecast(self):
        self.assertTrue(
            CHECKPOINT_PATH.is_file(),
            f"Trained checkpoint not found: {CHECKPOINT_PATH}",
        )

        service = MarineInferenceService(CHECKPOINT_PATH)
        request = make_request()

        result = service.predict(request)
        result.validate_against(request)

        self.assertEqual(result.model_version, "fno-0.2.0")
        self.assertEqual(result.contract_version, CONTRACT_VERSION)
        self.assertEqual(result.status, "success")

        for field in (
            result.water_level_m,
            result.current_u_ms,
            result.current_v_ms,
        ):
            self.assertEqual(len(field), 6)

            for step in field:
                self.assertEqual(len(step), request.grid.ny)

                for row in step:
                    self.assertEqual(len(row), request.grid.nx)
                    self.assertTrue(all(math.isfinite(value) for value in row))

        self.assertIsNotNone(service.target_mean)
        self.assertIsNotNone(service.target_std)


class TrainedMarineFNOAPITests(APITestCase):
    """Verify trained-model inference through the prediction API."""

    def setUp(self):
        super().setUp()

        self.assertTrue(
            CHECKPOINT_PATH.is_file(),
            f"Trained checkpoint not found: {CHECKPOINT_PATH}",
        )

        self.settings_override = override_settings(
            MARINE_AI_CHECKPOINT=str(CHECKPOINT_PATH),
            MARINE_AI_BACKEND="local",
            MARINE_AI_MODELARTS_ENDPOINT="",
            MARINE_AI_MODELARTS_TOKEN="",
            MARINE_AI_MODELARTS_TIMEOUT=30.0,
            MARINE_AI_MODELARTS_VERSION="modelarts-test",
        )
        self.settings_override.enable()
        self.addCleanup(self.settings_override.disable)

        get_marine_inference_service.cache_clear()
        self.addCleanup(get_marine_inference_service.cache_clear)

    def test_trained_checkpoint_prediction_is_persisted_through_api(self):
        response = self.client.post(
            "/api/v1/marine-predictions/create/",
            make_api_payload(),
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(MarinePrediction.objects.count(), 1)

        prediction = MarinePrediction.objects.get()
        result = prediction.result_data

        self.assertEqual(prediction.model_version, "fno-0.2.0")
        self.assertEqual(prediction.contract_version, CONTRACT_VERSION)
        self.assertEqual(
            prediction.status,
            MarinePrediction.Status.COMPLETED,
        )

        self.assertEqual(response.data["model_version"], "fno-0.2.0")
        self.assertEqual(response.data["contract_version"], CONTRACT_VERSION)

        for field_name in (
            "water_level_m",
            "current_u_ms",
            "current_v_ms",
        ):
            field = result[field_name]

            self.assertEqual(len(field), 6)

            for step in field:
                self.assertEqual(len(step), 8)

                for row in step:
                    self.assertEqual(len(row), 8)
                    self.assertTrue(all(math.isfinite(value) for value in row))
