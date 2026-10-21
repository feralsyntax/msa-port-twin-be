"""Tests for the marine FNO inference service."""

import tempfile
from pathlib import Path

import torch
from django.test import SimpleTestCase

from port_twin_app.services.marine_ai.contracts import (
    MarineForecastConfig,
    MarineGrid,
    MarinePredictionRequest,
)
from port_twin_app.services.marine_ai.inference import (
    MarineInferenceService,
)
from port_twin_app.services.marine_ai.model import (
    MODEL_VERSION,
    MarineFNO,
    save_checkpoint,
)


def make_request():
    """Create a small synthetic marine prediction request."""

    steps, ny, nx = 2, 8, 8

    return MarinePredictionRequest(
        grid=MarineGrid(
            west=39.5,
            south=-4.2,
            east=39.8,
            north=-3.9,
            nx=nx,
            ny=ny,
        ),
        forecast=MarineForecastConfig(
            start_time="2026-01-01T00:00:00+00:00",
            steps=steps,
            step_seconds=3600,
        ),
        bathymetry_m=[[20.0 for _ in range(nx)] for _ in range(ny)],
        initial_water_level_m=[[0.5 for _ in range(nx)] for _ in range(ny)],
        wind_u_ms=[[[2.0 for _ in range(nx)] for _ in range(ny)] for _ in range(steps)],
        wind_v_ms=[[[1.0 for _ in range(nx)] for _ in range(ny)] for _ in range(steps)],
        boundary_conditions={
            "west": {"water_level_m": [0.1] * steps},
        },
    )


class MarineInferenceServiceTests(SimpleTestCase):
    def setUp(self):
        torch.manual_seed(7)

        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)

        self.checkpoint_path = Path(self.temp_directory.name) / "marine_fno.pt"

        self.model = MarineFNO(
            width=4,
            modes_x=2,
            modes_y=2,
            layers=1,
        )

        save_checkpoint(
            self.model,
            self.checkpoint_path,
            epoch=1,
        )

        self.request = make_request()

    def _save_checkpoint_with_statistics(self):
        """Add target normalization statistics to a checkpoint."""

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        checkpoint["target_mean"] = torch.tensor(
            [[[[[0.5]], [[0.2]], [[0.3]]]]],
            dtype=torch.float32,
        )

        checkpoint["target_std"] = torch.tensor(
            [[[[[2.0]], [[3.0]], [[4.0]]]]],
            dtype=torch.float32,
        )

        torch.save(checkpoint, self.checkpoint_path)

        return checkpoint

    def test_loads_checkpoint_and_returns_valid_prediction(self):
        service = MarineInferenceService(self.checkpoint_path)

        result = service.predict(self.request)

        result.validate_against(self.request)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.model_version, MODEL_VERSION)
        self.assertEqual(len(result.water_level_m), 2)
        self.assertEqual(len(result.current_u_ms), 2)
        self.assertEqual(len(result.current_v_ms), 2)

        for field in (
            result.water_level_m,
            result.current_u_ms,
            result.current_v_ms,
        ):
            for step in field:
                self.assertEqual(len(step), self.request.grid.ny)
                for row in step:
                    self.assertEqual(len(row), self.request.grid.nx)

    def test_reuses_service_for_multiple_requests(self):
        service = MarineInferenceService(self.checkpoint_path)

        loaded_model = service.model

        first_result = service.predict(self.request)
        second_result = service.predict(self.request)

        self.assertIs(service.model, loaded_model)
        self.assertEqual(first_result.model_version, second_result.model_version)
        self.assertEqual(
            first_result.water_level_m,
            second_result.water_level_m,
        )
        self.assertEqual(
            first_result.current_u_ms,
            second_result.current_u_ms,
        )
        self.assertEqual(
            first_result.current_v_ms,
            second_result.current_v_ms,
        )

    def test_uses_saved_normalization_statistics(self):
        checkpoint = self._save_checkpoint_with_statistics()

        service = MarineInferenceService(self.checkpoint_path)

        self.assertTrue(
            torch.equal(
                service.target_mean,
                checkpoint["target_mean"],
            )
        )
        self.assertTrue(
            torch.equal(
                service.target_std,
                checkpoint["target_std"],
            )
        )

        result = service.predict(self.request)
        result.validate_against(self.request)

        self.assertEqual(result.model_version, checkpoint["model_version"])

    def test_missing_checkpoint_raises_loading_error(self):
        missing_path = Path(self.temp_directory.name) / "missing.pt"

        with self.assertRaises(FileNotFoundError):
            MarineInferenceService(missing_path)

    def test_rejects_incomplete_normalization_statistics(self):
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        checkpoint["target_mean"] = torch.zeros(
            (1, 1, 3, 1, 1),
            dtype=torch.float32,
        )
        # Deliberately omit target_std.
        torch.save(checkpoint, self.checkpoint_path)

        with self.assertRaisesMessage(
            ValueError,
            "Checkpoint must contain both target_mean and target_std.",
        ):
            MarineInferenceService(self.checkpoint_path)

    def test_rejects_invalid_request_type(self):
        service = MarineInferenceService(self.checkpoint_path)

        with self.assertRaisesMessage(
            TypeError,
            "request must be a MarinePredictionRequest instance.",
        ):
            service.predict({"invalid": "request"})
