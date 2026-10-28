"""HTTP adapter for a configurable remote marine inference endpoint."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .contracts import (
    CONTRACT_VERSION,
    MarinePredictionRequest,
    MarinePredictionResult,
)


class ModelArtsInferenceError(RuntimeError):
    """Raised when remote inference fails or returns an invalid response."""


class ModelArtsInferenceService:
    """Send marine prediction requests to a configured HTTP endpoint."""

    def __init__(
        self,
        endpoint_url: str,
        token: str = "",
        timeout: float = 30.0,
        model_version: str = "modelarts-unknown",
    ):
        if not endpoint_url.strip():
            raise ValueError("ModelArts endpoint URL must not be empty.")

        if timeout <= 0:
            raise ValueError("ModelArts timeout must be positive.")

        self.endpoint_url = endpoint_url
        self.token = token
        self.timeout = timeout
        self.model_version = model_version

    def predict(
        self,
        request: MarinePredictionRequest,
    ) -> MarinePredictionResult:
        """Submit a request and validate the returned prediction fields."""
        if not isinstance(request, MarinePredictionRequest):
            raise TypeError("request must be a MarinePredictionRequest instance.")

        payload = {
            "contract_version": CONTRACT_VERSION,
            "request": {
                "grid": {
                    "west": request.grid.west,
                    "south": request.grid.south,
                    "east": request.grid.east,
                    "north": request.grid.north,
                    "nx": request.grid.nx,
                    "ny": request.grid.ny,
                    "crs": request.grid.crs,
                },
                "forecast": {
                    "start_time": request.forecast.start_time,
                    "steps": request.forecast.steps,
                    "step_seconds": request.forecast.step_seconds,
                },
                "bathymetry_m": request.bathymetry_m,
                "initial_water_level_m": request.initial_water_level_m,
                "wind_u_ms": request.wind_u_ms,
                "wind_v_ms": request.wind_v_ms,
                "boundary_conditions": request.boundary_conditions,
            },
        }

        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        http_request = Request(
            self.endpoint_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urlopen(http_request, timeout=self.timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise ModelArtsInferenceError(
                "Remote marine inference endpoint request failed."
            ) from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelArtsInferenceError(
                "Remote inference endpoint returned invalid JSON."
            ) from exc

        if not isinstance(response_data, dict):
            raise ModelArtsInferenceError(
                "Remote inference response must be a JSON object."
            )

        try:
            result = MarinePredictionResult(
                water_level_m=response_data["water_level_m"],
                current_u_ms=response_data["current_u_ms"],
                current_v_ms=response_data["current_v_ms"],
                model_version=response_data.get("model_version", self.model_version),
                status=response_data.get("status", "success"),
                contract_version=response_data.get(
                    "contract_version", CONTRACT_VERSION
                ),
            )
            result.validate_against(request)
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelArtsInferenceError(
                "Remote inference response does not match the marine "
                "prediction contract."
            ) from exc

        return result
