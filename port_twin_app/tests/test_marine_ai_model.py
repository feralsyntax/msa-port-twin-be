"""
Tests for the initial marine Fourier Neural Operator.
"""

import tempfile
from pathlib import Path

import torch
from django.test import SimpleTestCase

from port_twin_app.services.marine_ai.contracts import (
    MarineForecastConfig,
    MarineGrid,
    MarinePredictionRequest,
)
from port_twin_app.services.marine_ai.model import (
    INPUT_CHANNELS,
    MODEL_VERSION,
    MarineFNO,
    load_checkpoint,
    predict,
    request_to_tensor,
    save_checkpoint,
)


def make_request(steps=3, ny=8, nx=10):
    """
    Build a small, valid request for model tests.
    """

    grid = MarineGrid(
        west=39.5,
        south=-4.2,
        east=39.8,
        north=-3.9,
        nx=nx,
        ny=ny,
    )
    forecast = MarineForecastConfig(
        start_time="2026-10-20T12:00:00+03:00",
        steps=steps,
        step_seconds=3600,
    )

    return MarinePredictionRequest(
        grid=grid,
        forecast=forecast,
        bathymetry_m=[[20.0 for _ in range(nx)] for _ in range(ny)],
        initial_water_level_m=[[0.5 for _ in range(nx)] for _ in range(ny)],
        wind_u_ms=[[[2.0 for _ in range(nx)] for _ in range(ny)] for _ in range(steps)],
        wind_v_ms=[[[1.0 for _ in range(nx)] for _ in range(ny)] for _ in range(steps)],
        boundary_conditions={
            "west": {"water_level_m": [0.1] * steps},
            "north": {"normal_velocity_ms": [0.2] * steps},
        },
    )


class MarineFNOTests(SimpleTestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.request = make_request()
        self.model = MarineFNO(
            width=8,
            modes_x=3,
            modes_y=3,
            layers=2,
        )

    def test_request_encoding_has_expected_shape(self):
        inputs = request_to_tensor(self.request)

        self.assertEqual(
            tuple(inputs.shape),
            (1, 3, INPUT_CHANNELS, 8, 10),
        )
        self.assertTrue(torch.isfinite(inputs).all())

    def test_request_encoding_includes_boundary_values(self):
        inputs = request_to_tensor(self.request)

        # Channels 7 and 8 are the west-edge water level and
        # east-edge normal velocity respectively, per the encoding.
        west_water_level_channel = 7
        north_normal_velocity_channel = 14

        self.assertTrue(torch.all(inputs[0, :, west_water_level_channel, :, 0] == 0.1))
        self.assertTrue(
            torch.all(inputs[0, :, north_normal_velocity_channel, -1, :] == 0.2)
        )

    def test_forward_pass_returns_three_fields_per_step(self):
        inputs = request_to_tensor(self.request)
        predictions = self.model(inputs)

        self.assertEqual(tuple(predictions.shape), (1, 3, 3, 8, 10))
        self.assertTrue(torch.isfinite(predictions).all())

    def test_model_supports_optimizer_update(self):
        inputs = request_to_tensor(self.request)
        targets = torch.zeros((1, 3, 3, 8, 10))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        before = [parameter.detach().clone() for parameter in self.model.parameters()]

        optimizer.zero_grad()
        predictions = self.model(inputs)
        loss = torch.nn.functional.mse_loss(predictions, targets)
        loss.backward()

        self.assertTrue(
            all(parameter.grad is not None for parameter in self.model.parameters())
        )

        optimizer.step()

        self.assertTrue(
            any(
                not torch.equal(old, new.detach())
                for old, new in zip(before, self.model.parameters())
            )
        )

    def test_prediction_matches_contract(self):
        result = predict(self.request, self.model)

        self.assertEqual(result.model_version, MODEL_VERSION)
        self.assertEqual(result.status, "success")
        self.assertEqual(len(result.water_level_m), 3)
        self.assertEqual(len(result.current_u_ms), 3)
        self.assertEqual(len(result.current_v_ms), 3)
        self.assertEqual(len(result.water_level_m[0]), 8)
        self.assertEqual(len(result.water_level_m[0][0]), 10)

        result.validate_against(self.request)

    def test_checkpoint_round_trip_preserves_predictions(self):
        inputs = request_to_tensor(self.request)
        self.model.eval()

        with torch.no_grad():
            expected = self.model(inputs)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "marine_fno.pt"
            save_checkpoint(self.model, checkpoint_path, epoch=1)
            restored_model, checkpoint = load_checkpoint(checkpoint_path)

        restored_model.eval()
        with torch.no_grad():
            actual = restored_model(inputs)

        self.assertEqual(checkpoint["epoch"], 1)
        self.assertEqual(checkpoint["model_version"], MODEL_VERSION)
        torch.testing.assert_close(actual, expected)
