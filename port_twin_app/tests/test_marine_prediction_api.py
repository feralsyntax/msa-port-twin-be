import tempfile
from pathlib import Path
from unittest.mock import patch

from django.test import override_settings
from rest_framework import status
from rest_framework.test import APITestCase

from port_twin_app.models import MarinePrediction
from port_twin_app.services.marine_ai.django_service import (
    get_marine_inference_service,
)
from port_twin_app.services.marine_ai.input_preparation import (
    MarineInputUnavailable,
    PreparedMarineInputs,
)
from port_twin_app.tests.test_marine_ai_inference import (
    create_test_checkpoint,
    make_request,
)


def make_api_payload(request=None):
    """Build the simplified public marine prediction payload."""
    if request is None:
        request = make_request()

    return {
        "forecast": {
            "start_time": request.forecast.start_time,
        },
    }


def make_prepared_inputs():
    """Build deterministic assembled inputs for API tests."""
    request = make_request()

    grid_data = {
        "west": request.grid.west,
        "south": request.grid.south,
        "east": request.grid.east,
        "north": request.grid.north,
        "nx": request.grid.nx,
        "ny": request.grid.ny,
        "crs": request.grid.crs,
        "domain": "mombasa_offshore",
    }

    forecast_data = {
        "start_time": request.forecast.start_time,
        "steps": request.forecast.steps,
        "step_seconds": request.forecast.step_seconds,
    }

    input_data = {
        "domain": "mombasa_offshore",
        "grid": grid_data,
        "source_data": {
            "marine": {
                "provider": "Copernicus Marine",
                "dataset_id": "test-copernicus-dataset",
                "dataset_version": "test",
                "selected_time": request.forecast.start_time,
            },
            "bathymetry": {
                "provider": "GEBCO",
                "dataset_id": "test-gebco-dataset",
                "dataset_version": "GEBCO_2026",
            },
        },
    }

    return PreparedMarineInputs(
        prediction_request=request,
        grid_data=grid_data,
        forecast_data=forecast_data,
        input_data=input_data,
    )


class MarinePredictionAPITests(APITestCase):
    def setUp(self):
        super().setUp()

        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        self.checkpoint_path = create_test_checkpoint(self.temp_dir.name)

        self.settings_override = override_settings(
            MARINE_AI_CHECKPOINT=str(self.checkpoint_path),
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

        self.list_url = "/api/v1/marine-predictions/"
        self.create_url = "/api/v1/marine-predictions/create/"

        self.prepared_inputs = make_prepared_inputs()

        # API tests use controlled assembled inputs rather than depending
        # on the freshness of the developer's local marine datasets.
        self.input_preparation_patcher = patch(
            "port_twin_app.views.prepare_marine_prediction_inputs",
            return_value=self.prepared_inputs,
        )
        self.mock_prepare_inputs = self.input_preparation_patcher.start()
        self.addCleanup(self.input_preparation_patcher.stop)

    def test_valid_request_creates_and_stores_prediction(self):
        response = self.client.post(
            self.create_url,
            make_api_payload(),
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(MarinePrediction.objects.count(), 1)

        prediction = MarinePrediction.objects.get()

        self.assertEqual(
            prediction.status,
            MarinePrediction.Status.COMPLETED,
        )
        self.assertEqual(str(prediction.id), response.data["id"])

        self.mock_prepare_inputs.assert_called_once()

    def test_public_request_does_not_require_internal_fno_arrays(self):
        response = self.client.post(
            self.create_url,
            make_api_payload(),
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

    def test_default_domain_is_passed_to_input_preparation(self):
        response = self.client.post(
            self.create_url,
            make_api_payload(),
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        self.mock_prepare_inputs.assert_called_once()
        self.assertEqual(
            self.mock_prepare_inputs.call_args.kwargs["domain"],
            "mombasa_offshore",
        )

    def test_response_contains_forecast_fields_and_model_version(self):
        response = self.client.post(
            self.create_url,
            make_api_payload(),
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        result = response.data["result_data"]

        self.assertIn("water_level_m", result)
        self.assertIn("current_u_ms", result)
        self.assertIn("current_v_ms", result)
        self.assertTrue(response.data["model_version"])

    def test_prediction_persists_source_provenance(self):
        response = self.client.post(
            self.create_url,
            make_api_payload(),
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        prediction = MarinePrediction.objects.get()

        self.assertEqual(
            prediction.input_data["domain"],
            "mombasa_offshore",
        )
        self.assertEqual(
            prediction.input_data["source_data"]["marine"]["provider"],
            "Copernicus Marine",
        )
        self.assertEqual(
            prediction.input_data["source_data"]["bathymetry"]["provider"],
            "GEBCO",
        )
        self.assertEqual(
            prediction.grid["domain"],
            "mombasa_offshore",
        )

    def test_missing_forecast_returns_400(self):
        response = self.client.post(
            self.create_url,
            {},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(MarinePrediction.objects.count(), 0)
        self.mock_prepare_inputs.assert_not_called()

    def test_missing_start_time_returns_400(self):
        response = self.client.post(
            self.create_url,
            {"forecast": {}},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(MarinePrediction.objects.count(), 0)
        self.mock_prepare_inputs.assert_not_called()

    def test_unsupported_domain_returns_400(self):
        payload = make_api_payload()
        payload["domain"] = "mombasa_harbour"

        response = self.client.post(
            self.create_url,
            payload,
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(MarinePrediction.objects.count(), 0)
        self.mock_prepare_inputs.assert_not_called()

    def test_missing_marine_data_returns_503(self):
        self.mock_prepare_inputs.side_effect = MarineInputUnavailable(
            "Copernicus dataset is missing."
        )

        response = self.client.post(
            self.create_url,
            make_api_payload(),
            format="json",
        )

        self.assertEqual(
            response.status_code,
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )
        self.assertEqual(MarinePrediction.objects.count(), 0)

    def test_stale_marine_data_returns_503(self):
        self.mock_prepare_inputs.side_effect = MarineInputUnavailable(
            "Copernicus dataset is stale."
        )

        response = self.client.post(
            self.create_url,
            make_api_payload(),
            format="json",
        )

        self.assertEqual(
            response.status_code,
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )
        self.assertEqual(MarinePrediction.objects.count(), 0)

    def test_missing_checkpoint_returns_503(self):
        missing_path = Path(self.temp_dir.name) / "missing.pt"

        with override_settings(MARINE_AI_CHECKPOINT=str(missing_path)):
            get_marine_inference_service.cache_clear()

            response = self.client.post(
                self.create_url,
                make_api_payload(),
                format="json",
            )

        self.assertEqual(
            response.status_code,
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )
        self.assertEqual(MarinePrediction.objects.count(), 0)

    def test_list_endpoint_includes_stored_predictions(self):
        create_response = self.client.post(
            self.create_url,
            make_api_payload(),
            format="json",
        )
        self.assertEqual(
            create_response.status_code,
            status.HTTP_201_CREATED,
        )

        response = self.client.get(self.list_url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)

        results = (
            response.data["results"]
            if isinstance(response.data, dict) and "results" in response.data
            else response.data
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], create_response.data["id"])

    def test_detail_endpoint_retrieves_prediction_by_uuid(self):
        create_response = self.client.post(
            self.create_url,
            make_api_payload(),
            format="json",
        )
        self.assertEqual(
            create_response.status_code,
            status.HTTP_201_CREATED,
        )

        prediction_id = create_response.data["id"]
        response = self.client.get(f"/api/v1/marine-predictions/{prediction_id}/")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["id"], prediction_id)

    @patch("port_twin_app.views.get_configured_marine_inference_service")
    def test_inference_runtime_failure_returns_503_without_persisting(
        self,
        mock_get_service,
    ):
        mock_service = mock_get_service.return_value
        mock_service.predict.side_effect = RuntimeError("Simulated inference failure")

        response = self.client.post(
            self.create_url,
            make_api_payload(),
            format="json",
        )

        self.assertEqual(
            response.status_code,
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )
        self.assertEqual(MarinePrediction.objects.count(), 0)
        mock_service.predict.assert_called_once()

    @patch(
        "port_twin_app.models.MarinePrediction.save",
        side_effect=RuntimeError("Simulated database failure"),
    )
    def test_database_persistence_failure_returns_500_without_persisting(
        self,
        mock_save,
    ):
        response = self.client.post(
            self.create_url,
            make_api_payload(),
            format="json",
        )

        self.assertEqual(
            response.status_code,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
        self.assertEqual(MarinePrediction.objects.count(), 0)
        mock_save.assert_called_once()
