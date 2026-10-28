import json
from unittest.mock import MagicMock, patch
from urllib.error import URLError

from django.test import SimpleTestCase

from port_twin_app.services.marine_ai.contracts import (
    CONTRACT_VERSION,
    MarinePredictionResult,
)
from port_twin_app.services.marine_ai.modelarts import (
    ModelArtsInferenceError,
    ModelArtsInferenceService,
)
from port_twin_app.tests.test_marine_ai_inference import make_request


class ModelArtsInferenceServiceTests(SimpleTestCase):
    def setUp(self):
        self.prediction_request = make_request()
        self.service = ModelArtsInferenceService(
            endpoint_url="https://example.com/inference",
            token="test-token",
            timeout=10.0,
            model_version="modelarts-test",
        )

    @patch("port_twin_app.services.marine_ai.modelarts.urlopen")
    def test_successful_prediction_sends_contract_and_parses_response(
        self, mock_urlopen
    ):
        # Match the output dimensions to the supplied prediction request.
        steps = self.prediction_request.forecast.steps
        ny = self.prediction_request.grid.ny
        nx = self.prediction_request.grid.nx

        water_level_m = [
            [[0.1 for _ in range(nx)] for _ in range(ny)] for _ in range(steps)
        ]
        current_u_ms = [
            [[1.0 for _ in range(nx)] for _ in range(ny)] for _ in range(steps)
        ]
        current_v_ms = [
            [[2.0 for _ in range(nx)] for _ in range(ny)] for _ in range(steps)
        ]

        expected = MarinePredictionResult(
            water_level_m=water_level_m,
            current_u_ms=current_u_ms,
            current_v_ms=current_v_ms,
            model_version="modelarts-test",
            status="success",
            contract_version=CONTRACT_VERSION,
        )

        response = MagicMock()
        response.read.return_value = json.dumps(
            {
                "water_level_m": expected.water_level_m,
                "current_u_ms": expected.current_u_ms,
                "current_v_ms": expected.current_v_ms,
                "model_version": expected.model_version,
                "status": expected.status,
                "contract_version": expected.contract_version,
            }
        ).encode("utf-8")
        response.__enter__.return_value = response
        mock_urlopen.return_value = response

        result = self.service.predict(self.prediction_request)

        self.assertEqual(result, expected)
        mock_urlopen.assert_called_once()

        http_request = mock_urlopen.call_args.args[0]
        self.assertEqual(http_request.method, "POST")
        self.assertEqual(
            http_request.get_header("Authorization"),
            "Bearer test-token",
        )

        payload = json.loads(http_request.data.decode("utf-8"))
        self.assertEqual(payload["contract_version"], CONTRACT_VERSION)
        self.assertIn("grid", payload["request"])
        self.assertIn("forecast", payload["request"])

        self.assertEqual(
            mock_urlopen.call_args.kwargs["timeout"],
            10.0,
        )

    @patch("port_twin_app.services.marine_ai.modelarts.urlopen")
    def test_http_failure_raises_modelarts_error(self, mock_urlopen):
        mock_urlopen.side_effect = URLError("connection failed")

        with self.assertRaises(ModelArtsInferenceError):
            self.service.predict(self.prediction_request)

    @patch("port_twin_app.services.marine_ai.modelarts.urlopen")
    def test_invalid_json_raises_modelarts_error(self, mock_urlopen):
        response = MagicMock()
        response.read.return_value = b"not-json"
        response.__enter__.return_value = response
        mock_urlopen.return_value = response

        with self.assertRaises(ModelArtsInferenceError):
            self.service.predict(self.prediction_request)

    @patch("port_twin_app.services.marine_ai.modelarts.urlopen")
    def test_invalid_prediction_shape_raises_modelarts_error(self, mock_urlopen):
        response = MagicMock()
        response.read.return_value = json.dumps(
            {
                "water_level_m": [[[0.1]]],
                "current_u_ms": [[[0.2]]],
                "current_v_ms": [[[0.3]]],
            }
        ).encode("utf-8")
        response.__enter__.return_value = response
        mock_urlopen.return_value = response

        with self.assertRaises(ModelArtsInferenceError):
            self.service.predict(self.prediction_request)

    def test_empty_endpoint_is_rejected(self):
        with self.assertRaises(ValueError):
            ModelArtsInferenceService(endpoint_url="")

    def test_nonpositive_timeout_is_rejected(self):
        with self.assertRaises(ValueError):
            ModelArtsInferenceService(
                endpoint_url="https://example.com/inference",
                timeout=0,
            )
