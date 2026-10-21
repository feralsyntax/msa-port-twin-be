import tempfile
from pathlib import Path

from django.test import override_settings
from rest_framework import status
from rest_framework.test import APITestCase

from port_twin_app.models import MarinePrediction
from port_twin_app.services.marine_ai.django_service import (
    get_marine_inference_service,
)
from port_twin_app.tests.test_marine_ai_inference import (
    create_test_checkpoint,
    make_request,
)


def make_api_payload():
    """Convert the existing inference request fixture into JSON input."""
    request = make_request()
    grid = request.grid
    forecast = request.forecast

    return {
        "grid": {
            "west": grid.west,
            "south": grid.south,
            "east": grid.east,
            "north": grid.north,
            "nx": grid.nx,
            "ny": grid.ny,
            "crs": grid.crs,
        },
        "forecast": {
            "start_time": forecast.start_time.isoformat(),
            "steps": forecast.steps,
            "step_seconds": forecast.step_seconds,
        },
        "bathymetry_m": request.bathymetry_m,
        "initial_water_level_m": request.initial_water_level_m,
        "wind_u_ms": request.wind_u_ms,
        "wind_v_ms": request.wind_v_ms,
        "boundary_conditions": request.boundary_conditions,
    }


class MarinePredictionAPITests(APITestCase):
    def setUp(self):
        super().setUp()

        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        self.checkpoint_path = create_test_checkpoint(self.temp_dir.name)

        self.settings_override = override_settings(
            MARINE_AI_CHECKPOINT=str(self.checkpoint_path)
        )
        self.settings_override.enable()
        self.addCleanup(self.settings_override.disable)

        # Prevent a cached service from using another test's checkpoint.
        get_marine_inference_service.cache_clear()
        self.addCleanup(get_marine_inference_service.cache_clear)

        self.list_url = "/api/v1/marine-predictions/"
        self.create_url = "/api/v1/marine-predictions/create/"

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

    def test_invalid_grid_dimensions_return_400(self):
        payload = make_api_payload()
        payload["grid"]["nx"] = 1

        response = self.client.post(
            self.create_url,
            payload,
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(MarinePrediction.objects.count(), 0)

    def test_malformed_request_returns_400(self):
        response = self.client.post(
            self.create_url,
            {"grid": {}, "forecast": {}},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
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
