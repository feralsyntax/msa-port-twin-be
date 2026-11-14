"""Integration tests verifying inference with the trained marine FNO artifact."""

import math
from pathlib import Path
from unittest.mock import patch

from django.test import SimpleTestCase, override_settings
from rest_framework import status
from rest_framework.test import APITestCase

from port_twin_app.models import MarinePrediction
from port_twin_app.services.marine_ai.contracts import CONTRACT_VERSION
from port_twin_app.services.marine_ai.django_service import (
    get_marine_inference_service,
)
from port_twin_app.services.marine_ai.inference import (
    MarineInferenceService,
)
from port_twin_app.services.marine_ai.input_preparation import (
    PreparedMarineInputs,
)
from port_twin_app.services.marine_ai.model import (
    PREDICTION_MODE_RESIDUAL,
    RESIDUAL_MODEL_VERSION,
)
from port_twin_app.services.marine_ai.serialization import (
    load_prepared_dataset,
)
from port_twin_app.tests.test_marine_ai_inference import make_request
from port_twin_app.tests.test_marine_prediction_api import make_api_payload

CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[2] / "artifacts" / "marine_fno_residual.pt"
)


def make_prepared_inputs(request):
    """Build prepared inputs around an existing internal FNO test request."""

    grid = request.grid
    forecast = request.forecast

    return PreparedMarineInputs(
        prediction_request=request,
        grid_data={
            "west": grid.west,
            "south": grid.south,
            "east": grid.east,
            "north": grid.north,
            "nx": grid.nx,
            "ny": grid.ny,
            "crs": grid.crs,
            "domain": "mombasa_offshore",
        },
        forecast_data={
            "start_time": forecast.start_time,
            "steps": forecast.steps,
            "step_seconds": forecast.step_seconds,
        },
        input_data={
            "domain": "mombasa_offshore",
            "test_fixture": True,
            "source": "held_out_test_sample",
        },
    )


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

        self.assertEqual(result.model_version, RESIDUAL_MODEL_VERSION)
        self.assertEqual(service.model_version, RESIDUAL_MODEL_VERSION)
        self.assertEqual(service.prediction_mode, PREDICTION_MODE_RESIDUAL)
        self.assertEqual(service.contract_version, CONTRACT_VERSION)
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
        request = make_request()
        prepared_inputs = make_prepared_inputs(request)

        with patch(
            "port_twin_app.views.prepare_marine_prediction_inputs",
            return_value=prepared_inputs,
        ):
            response = self.client.post(
                "/api/v1/marine-predictions/create/",
                make_api_payload(request),
                format="json",
            )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(MarinePrediction.objects.count(), 1)

        prediction = MarinePrediction.objects.get()
        result = prediction.result_data

        self.assertEqual(prediction.model_version, RESIDUAL_MODEL_VERSION)
        self.assertEqual(prediction.contract_version, CONTRACT_VERSION)
        self.assertEqual(
            prediction.status,
            MarinePrediction.Status.COMPLETED,
        )

        self.assertEqual(response.data["model_version"], RESIDUAL_MODEL_VERSION)
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

    def test_real_dataset_prediction_is_persisted_and_retrievable(self):
        """Verify API inference using an actual held-out Mombasa sample."""

        dataset_path = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "processed"
            / "marine"
            / "mombasa_training_dataset.npz"
        )

        self.assertTrue(
            dataset_path.is_file(),
            f"Prepared marine dataset not found: {dataset_path}",
        )

        prepared = load_prepared_dataset(dataset_path)
        sample = prepared.split.test[0]
        request = sample.request

        with override_settings(MARINE_DATA_MAX_AGE_HOURS=24 * 30):
            response = self.client.post(
                "/api/v1/marine-predictions/create/",
                make_api_payload(request),
                format="json",
            )

        self.assertEqual(
            response.status_code,
            status.HTTP_201_CREATED,
            response.data,
        )
        self.assertEqual(MarinePrediction.objects.count(), 1)

        prediction = MarinePrediction.objects.get()

        self.assertEqual(prediction.grid["nx"], 10)
        self.assertEqual(prediction.grid["ny"], 23)
        self.assertNotIn("test_fixture", prediction.input_data)
        self.assertIn("marine", prediction.input_data["source_data"])
        self.assertIn("bathymetry", prediction.input_data["source_data"])

        self.assertEqual(prediction.model_version, RESIDUAL_MODEL_VERSION)
        self.assertEqual(prediction.contract_version, CONTRACT_VERSION)
        self.assertEqual(
            prediction.status,
            MarinePrediction.Status.COMPLETED,
        )

        self.assertEqual(response.data["id"], str(prediction.pk))
        self.assertEqual(response.data["model_version"], RESIDUAL_MODEL_VERSION)
        self.assertEqual(response.data["contract_version"], CONTRACT_VERSION)

        result = prediction.result_data

        for field_name in (
            "water_level_m",
            "current_u_ms",
            "current_v_ms",
        ):
            field = result[field_name]

            self.assertEqual(len(field), 6)

            for step in field:
                self.assertEqual(len(step), prediction.grid["ny"])

                for row in step:
                    self.assertEqual(len(row), prediction.grid["nx"])
                    self.assertTrue(all(math.isfinite(value) for value in row))

        detail_response = self.client.get(
            f"/api/v1/marine-predictions/{prediction.pk}/"
        )

        self.assertEqual(
            detail_response.status_code,
            status.HTTP_200_OK,
        )
        self.assertEqual(
            detail_response.data["id"],
            str(prediction.pk),
        )
        self.assertEqual(
            detail_response.data["model_version"],
            RESIDUAL_MODEL_VERSION,
        )
        self.assertEqual(
            detail_response.data["result_data"],
            result,
        )
