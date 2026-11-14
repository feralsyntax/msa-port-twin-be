from unittest.mock import patch

from rest_framework import status
from rest_framework.test import APITestCase

from port_twin_app.models import MarinePrediction
from port_twin_app.services.marine_ai.input_preparation import (
    MarineInputUnavailable,
    PreparedMarineInputs,
)
from port_twin_app.services.marine_ai.prediction_execution import (
    SAFE_ERROR_MESSAGE,
    execute_marine_prediction,
)
from port_twin_app.tests.test_marine_ai_inference import make_request


def make_public_payload(request=None):
    """Build the simplified frontend-facing request."""

    if request is None:
        request = make_request()

    return {
        "forecast": {
            "start_time": request.forecast.start_time,
        },
        "domain": "mombasa_offshore",
    }


def make_api_payload(request=None):
    """Backward-compatible alias for the public API payload fixture."""
    return make_public_payload(request)


class MarinePredictionAPITests(APITestCase):
    def setUp(self):
        super().setUp()

        self.internal_request = make_request()

        grid = self.internal_request.grid
        forecast = self.internal_request.forecast

        self.prepared_inputs = PreparedMarineInputs(
            prediction_request=self.internal_request,
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
                "grid": {
                    "west": grid.west,
                    "south": grid.south,
                    "east": grid.east,
                    "north": grid.north,
                    "nx": grid.nx,
                    "ny": grid.ny,
                },
                "source_data": {
                    "marine": {
                        "provider": "Copernicus Marine",
                        "dataset_id": "test-copernicus",
                        "dataset_version": "test-version",
                        "selected_time": forecast.start_time,
                    },
                    "bathymetry": {
                        "provider": "GEBCO",
                        "dataset_id": "GEBCO",
                        "dataset_version": "GEBCO_2026",
                    },
                },
            },
        )

        self.preparation_patcher = patch(
            "port_twin_app.views.prepare_marine_prediction_inputs",
            return_value=self.prepared_inputs,
        )
        self.mock_prepare_inputs = self.preparation_patcher.start()
        self.addCleanup(self.preparation_patcher.stop)

        self.queue_patcher = patch(
            "port_twin_app.views.execute_marine_prediction_task.delay"
        )
        self.mock_queue_task = self.queue_patcher.start()
        self.addCleanup(self.queue_patcher.stop)

        self.list_url = "/api/v1/marine-predictions/"
        self.create_url = "/api/v1/marine-predictions/create/"
        self.status_url = "/api/v1/marine-predictions/{}/status/"

    def test_valid_request_creates_pending_prediction_and_queues_task(self):
        response = self.client.post(
            self.create_url,
            make_public_payload(self.internal_request),
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)
        self.assertEqual(MarinePrediction.objects.count(), 1)

        prediction = MarinePrediction.objects.get()

        self.assertEqual(
            prediction.status,
            MarinePrediction.Status.PENDING,
        )
        self.assertIsNone(prediction.result_data)
        self.assertEqual(str(prediction.id), response.data["id"])
        self.mock_queue_task.assert_called_once_with(str(prediction.id))
        self.mock_prepare_inputs.assert_called_once()

    def test_creation_response_contains_prediction_metadata(self):
        response = self.client.post(
            self.create_url,
            make_public_payload(self.internal_request),
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)
        self.assertIsNone(response.data["result_data"])
        self.assertEqual(response.data["status"], MarinePrediction.Status.PENDING)
        self.assertEqual(response.data["model_status"], "experimental")
        self.assertEqual(response.data["prediction_mode"], "residual")
        self.assertEqual(response.data["forecast_horizon_hours"], 6)
        self.assertTrue(response.data["model_version"])
        self.assertTrue(response.data["contract_version"])

    def test_public_request_does_not_require_internal_arrays(self):
        response = self.client.post(
            self.create_url,
            {
                "forecast": {
                    "start_time": self.internal_request.forecast.start_time,
                },
            },
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)
        self.mock_prepare_inputs.assert_called_once()
        self.assertEqual(
            self.mock_prepare_inputs.call_args.kwargs["domain"],
            "mombasa_offshore",
        )

    def test_unsupported_domain_returns_400(self):
        payload = make_public_payload(self.internal_request)
        payload["domain"] = "mombasa_harbour"

        response = self.client.post(
            self.create_url,
            payload,
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(MarinePrediction.objects.count(), 0)
        self.mock_prepare_inputs.assert_not_called()
        self.mock_queue_task.assert_not_called()

    def test_malformed_request_returns_400(self):
        response = self.client.post(
            self.create_url,
            {"forecast": {}},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(MarinePrediction.objects.count(), 0)
        self.mock_prepare_inputs.assert_not_called()
        self.mock_queue_task.assert_not_called()

    def test_missing_forecast_returns_400(self):
        response = self.client.post(
            self.create_url,
            {},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(MarinePrediction.objects.count(), 0)

    def test_stale_marine_data_returns_503_without_persisting(self):
        self.mock_prepare_inputs.side_effect = MarineInputUnavailable(
            "Copernicus dataset is stale."
        )

        response = self.client.post(
            self.create_url,
            make_public_payload(self.internal_request),
            format="json",
        )

        self.assertEqual(
            response.status_code,
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )
        self.assertEqual(MarinePrediction.objects.count(), 0)
        self.mock_queue_task.assert_not_called()

    def test_missing_marine_dataset_returns_503_without_persisting(self):
        self.mock_prepare_inputs.side_effect = MarineInputUnavailable(
            "Configured marine dataset does not exist."
        )

        response = self.client.post(
            self.create_url,
            make_public_payload(self.internal_request),
            format="json",
        )

        self.assertEqual(
            response.status_code,
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )
        self.assertEqual(MarinePrediction.objects.count(), 0)
        self.mock_queue_task.assert_not_called()

    def test_prediction_persists_validated_inputs_and_provenance(self):
        response = self.client.post(
            self.create_url,
            make_public_payload(self.internal_request),
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)

        prediction = MarinePrediction.objects.get()

        self.assertEqual(
            prediction.provenance["marine"]["provider"],
            "Copernicus Marine",
        )
        self.assertEqual(
            prediction.provenance["bathymetry"]["provider"],
            "GEBCO",
        )
        self.assertEqual(prediction.input_data["domain"], "mombasa_offshore")
        self.assertIn("validated_inputs", prediction.input_data)
        self.assertNotIn("source_data", prediction.input_data)

        validated_inputs = prediction.input_data["validated_inputs"]
        self.assertEqual(
            set(validated_inputs),
            {
                "bathymetry_m",
                "initial_water_level_m",
                "initial_current_u_ms",
                "initial_current_v_ms",
            },
        )

    def test_list_endpoint_includes_stored_predictions(self):
        create_response = self.client.post(
            self.create_url,
            make_public_payload(self.internal_request),
            format="json",
        )
        self.assertEqual(
            create_response.status_code,
            status.HTTP_202_ACCEPTED,
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
            make_public_payload(self.internal_request),
            format="json",
        )
        self.assertEqual(
            create_response.status_code,
            status.HTTP_202_ACCEPTED,
        )

        prediction_id = create_response.data["id"]
        response = self.client.get(f"/api/v1/marine-predictions/{prediction_id}/")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["id"], prediction_id)

    def test_status_endpoint_retrieves_prediction_status(self):
        create_response = self.client.post(
            self.create_url,
            make_public_payload(self.internal_request),
            format="json",
        )

        prediction_id = create_response.data["id"]
        response = self.client.get(self.status_url.format(prediction_id))

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["id"], prediction_id)
        self.assertEqual(
            response.data["status"],
            MarinePrediction.Status.PENDING,
        )
        self.assertIn("model_status", response.data)
        self.assertIn("prediction_mode", response.data)
        self.assertIn("forecast_horizon_hours", response.data)

    def test_inference_failure_persists_failed_prediction(self):
        self.client.post(
            self.create_url,
            make_public_payload(self.internal_request),
            format="json",
        )
        prediction = MarinePrediction.objects.get()

        with patch(
            "port_twin_app.services.marine_ai.prediction_execution."
            "get_configured_marine_inference_service"
        ) as mock_get_service:
            mock_get_service.return_value.predict.side_effect = RuntimeError(
                "Internal model failure details"
            )

            with self.assertRaises(RuntimeError):
                execute_marine_prediction(str(prediction.id))

        prediction.refresh_from_db()

        self.assertEqual(prediction.status, MarinePrediction.Status.FAILED)
        self.assertEqual(prediction.error_message, SAFE_ERROR_MESSAGE)
        self.assertNotIn(
            "Internal model failure details",
            prediction.error_message,
        )
        self.assertIsNone(prediction.result_data)

    def test_queue_failure_persists_failed_prediction(self):
        self.mock_queue_task.side_effect = RuntimeError("Broker unavailable")

        response = self.client.post(
            self.create_url,
            make_public_payload(self.internal_request),
            format="json",
        )

        self.assertEqual(
            response.status_code,
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )

        prediction = MarinePrediction.objects.get()
        self.assertEqual(prediction.status, MarinePrediction.Status.FAILED)
        self.assertEqual(
            prediction.error_message,
            "Marine prediction could not be queued. Please try again.",
        )

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
            make_public_payload(self.internal_request),
            format="json",
        )

        self.assertEqual(
            response.status_code,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
        self.assertEqual(MarinePrediction.objects.count(), 0)
        mock_save.assert_called_once()
