"""Tests for the marine FNO model and inference utilities."""

import tempfile
from pathlib import Path

import torch
from django.test import SimpleTestCase

from port_twin_app.services.marine_ai.contracts import (
    FORECAST_STEPS,
    MarineForecastConfig,
    MarineGrid,
    MarinePredictionRequest,
)
from port_twin_app.services.marine_ai.model import (
    CONTRACT_VERSION,
    INPUT_CHANNELS,
    MODEL_VERSION,
    PREDICTION_MODE_RESIDUAL,
    RESIDUAL_MODEL_VERSION,
    MarineFNO,
    SpectralConv2d,
    load_checkpoint,
    predict,
    predict_from_checkpoint,
    request_to_tensor,
    save_checkpoint,
)


def make_request(ny=8, nx=10):
    """Build a small request suitable for fast model tests."""

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
        ),
        bathymetry_m=[[20.0 for _ in range(nx)] for _ in range(ny)],
        initial_water_level_m=[[0.5 for _ in range(nx)] for _ in range(ny)],
        initial_current_u_ms=[[2.0 for _ in range(nx)] for _ in range(ny)],
        initial_current_v_ms=[[1.0 for _ in range(nx)] for _ in range(ny)],
    )


class MarineFNOModelTests(SimpleTestCase):
    def setUp(self):
        torch.manual_seed(7)

        self.request = make_request()
        self.model = MarineFNO(
            width=4,
            modes_x=2,
            modes_y=2,
            layers=1,
        )

    def _zero_model_parameters(self):
        """Make the model produce zero output."""

        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.zero_()

    def test_request_to_tensor_has_expected_shape(self):
        inputs = request_to_tensor(self.request)

        self.assertEqual(
            tuple(inputs.shape),
            (1, INPUT_CHANNELS, 8, 10),
        )
        self.assertTrue(torch.isfinite(inputs).all())

    def test_request_to_tensor_encodes_spatial_fields(self):
        inputs = request_to_tensor(self.request)

        self.assertTrue(torch.all(inputs[0, 0] == 20.0))
        self.assertTrue(torch.all(inputs[0, 1] == 0.5))
        self.assertTrue(torch.all(inputs[0, 2] == 2.0))
        self.assertTrue(torch.all(inputs[0, 3] == 1.0))

    def test_request_to_tensor_encodes_normalized_coordinates(self):
        inputs = request_to_tensor(self.request)

        longitude = inputs[0, 4]
        latitude = inputs[0, 5]

        self.assertAlmostEqual(longitude[0, 0].item(), 39.5 / 180.0, places=5)
        self.assertAlmostEqual(longitude[0, -1].item(), 39.8 / 180.0, places=5)
        self.assertAlmostEqual(latitude[0, 0].item(), -4.2 / 90.0, places=5)
        self.assertAlmostEqual(latitude[-1, 0].item(), -3.9 / 90.0, places=5)

    def test_request_to_tensor_rejects_invalid_grid_shape(self):
        request = make_request()
        object.__setattr__(request, "bathymetry_m", [[20.0, 30.0]])

        with self.assertRaisesRegex(ValueError, "bathymetry_m must have shape"):
            request_to_tensor(request)

    def test_request_to_tensor_rejects_non_finite_grid_values(self):
        self.request.bathymetry_m[0][0] = float("nan")

        with self.assertRaisesRegex(ValueError, "bathymetry_m contains non-finite"):
            request_to_tensor(self.request)

    def test_request_to_tensor_rejects_invalid_current_u_shape(self):
        request = make_request()
        object.__setattr__(request, "initial_current_u_ms", [])

        with self.assertRaisesRegex(
            ValueError,
            "initial_current_u_ms must have shape",
        ):
            request_to_tensor(request)

    def test_request_to_tensor_rejects_invalid_current_v_shape(self):
        request = make_request()
        object.__setattr__(request, "initial_current_v_ms", [])

        with self.assertRaisesRegex(
            ValueError,
            "initial_current_v_ms must have shape",
        ):
            request_to_tensor(request)

    def test_request_to_tensor_rejects_non_finite_current_values(self):
        self.request.initial_current_v_ms[0][0] = float("inf")

        with self.assertRaisesRegex(
            ValueError,
            "initial_current_v_ms contains non-finite",
        ):
            request_to_tensor(self.request)

    def test_forward_returns_expected_shape(self):
        inputs = request_to_tensor(self.request)
        outputs = self.model(inputs)

        self.assertEqual(
            tuple(outputs.shape),
            (1, FORECAST_STEPS, 3, 8, 10),
        )
        self.assertTrue(torch.isfinite(outputs).all())

    def test_optimizer_updates_model_parameters(self):
        inputs = request_to_tensor(self.request)
        before = [parameter.detach().clone() for parameter in self.model.parameters()]

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)

        outputs = self.model(inputs)
        loss = outputs.square().mean()
        loss.backward()
        optimizer.step()

        self.assertTrue(
            any(
                not torch.equal(old, new.detach())
                for old, new in zip(before, self.model.parameters())
            )
        )

    def test_predict_returns_valid_prediction_result(self):
        result = predict(self.request, self.model)

        result.validate_against(self.request)

        self.assertEqual(result.model_version, MODEL_VERSION)
        self.assertEqual(len(result.water_level_m), FORECAST_STEPS)
        self.assertEqual(len(result.current_u_ms), FORECAST_STEPS)
        self.assertEqual(len(result.current_v_ms), FORECAST_STEPS)
        self.assertEqual(len(result.water_level_m[0]), 8)
        self.assertEqual(len(result.water_level_m[0][0]), 10)

    def test_predict_reverses_target_normalization(self):
        mean = torch.tensor([0.5, 2.0, -1.0]).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.2, 0.4, 0.6]).view(1, 1, 3, 1, 1)

        self._zero_model_parameters()

        result = predict(
            self.request,
            self.model,
            target_mean=mean,
            target_std=std,
        )

        self.assertAlmostEqual(result.water_level_m[0][0][0], 0.5, places=5)
        self.assertAlmostEqual(result.current_u_ms[0][0][0], 2.0, places=5)
        self.assertAlmostEqual(result.current_v_ms[0][0][0], -1.0, places=5)

    def test_predict_residual_zero_output_reconstructs_initial_state(self):
        self._zero_model_parameters()

        result = predict(
            self.request,
            self.model,
            prediction_mode=PREDICTION_MODE_RESIDUAL,
        )

        result.validate_against(self.request)
        self.assertEqual(result.model_version, RESIDUAL_MODEL_VERSION)

        expected_fields = (
            (result.water_level_m, self.request.initial_water_level_m),
            (result.current_u_ms, self.request.initial_current_u_ms),
            (result.current_v_ms, self.request.initial_current_v_ms),
        )

        for predictions, initial_values in expected_fields:
            expected = torch.tensor(initial_values, dtype=torch.float32)

            for step in predictions:
                actual = torch.tensor(step, dtype=torch.float32)
                self.assertTrue(torch.equal(actual, expected))

    def test_predict_residual_denormalizes_before_adding_initial_state(self):
        mean = torch.tensor([0.1, 0.2, -0.3]).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.5, 2.0, 3.0]).view(1, 1, 3, 1, 1)

        self._zero_model_parameters()

        result = predict(
            self.request,
            self.model,
            target_mean=mean,
            target_std=std,
            prediction_mode=PREDICTION_MODE_RESIDUAL,
        )

        expected_values = (
            (0.5 + 0.1, 0.5),
            (2.0 + 0.2, 2.0),
            (1.0 - 0.3, 1.0),
        )

        actual_values = (
            result.water_level_m[0][0][0],
            result.current_u_ms[0][0][0],
            result.current_v_ms[0][0][0],
        )

        for actual, (expected, _) in zip(actual_values, expected_values):
            self.assertAlmostEqual(actual, expected, places=5)

    def test_predict_requires_both_normalization_statistics(self):
        mean = torch.zeros((1, 1, 3, 1, 1))

        with self.assertRaises(ValueError):
            predict(self.request, self.model, target_mean=mean)

        with self.assertRaises(ValueError):
            predict(self.request, self.model, target_std=torch.ones_like(mean))

    def test_predict_rejects_invalid_normalization_statistics(self):
        mean = torch.zeros((1, 1, 3, 1, 1))
        std = torch.ones((1, 1, 3, 1, 1))
        std[0, 0, 0, 0, 0] = 0.0

        with self.assertRaises(ValueError):
            predict(
                self.request,
                self.model,
                target_mean=mean,
                target_std=std,
            )

    def test_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "marine_fno.pt"

            save_checkpoint(self.model, checkpoint_path, epoch=3)
            restored_model, checkpoint = load_checkpoint(checkpoint_path)

            self.assertIsInstance(restored_model, MarineFNO)
            self.assertEqual(checkpoint["epoch"], 3)
            self.assertEqual(checkpoint["model_version"], MODEL_VERSION)
            self.assertEqual(checkpoint["architecture"]["width"], 4)
            self.assertEqual(
                checkpoint["architecture"]["forecast_steps"],
                FORECAST_STEPS,
            )

            original_output = self.model(request_to_tensor(self.request))
            restored_output = restored_model(request_to_tensor(self.request))

            self.assertTrue(torch.allclose(original_output, restored_output, atol=1e-6))

    def test_residual_checkpoint_round_trip_preserves_prediction_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "marine_fno_residual.pt"

            save_checkpoint(
                self.model,
                checkpoint_path,
                epoch=3,
                prediction_mode=PREDICTION_MODE_RESIDUAL,
            )

            restored_model, checkpoint = load_checkpoint(checkpoint_path)

            self.assertIsInstance(restored_model, MarineFNO)
            self.assertEqual(
                checkpoint["prediction_mode"],
                PREDICTION_MODE_RESIDUAL,
            )
            self.assertEqual(
                checkpoint["model_version"],
                RESIDUAL_MODEL_VERSION,
            )

            result = predict_from_checkpoint(
                self.request,
                checkpoint_path,
            )

            self.assertEqual(result.model_version, RESIDUAL_MODEL_VERSION)
            result.validate_against(self.request)

    def test_predict_from_checkpoint_uses_saved_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "marine_fno.pt"

            self._zero_model_parameters()

            save_checkpoint(self.model, checkpoint_path, epoch=2)

            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            checkpoint["target_mean"] = torch.tensor([0.5, 2.0, -1.0]).view(
                1, 1, 3, 1, 1
            )
            checkpoint["target_std"] = torch.tensor([0.2, 0.4, 0.6]).view(1, 1, 3, 1, 1)
            torch.save(checkpoint, checkpoint_path)

            result = predict_from_checkpoint(self.request, checkpoint_path)

            self.assertAlmostEqual(result.water_level_m[0][0][0], 0.5, places=5)
            self.assertAlmostEqual(result.current_u_ms[0][0][0], 2.0, places=5)
            self.assertAlmostEqual(result.current_v_ms[0][0][0], -1.0, places=5)

    def test_model_rejects_invalid_architecture_parameters(self):
        with self.assertRaisesRegex(ValueError, "width must be positive"):
            MarineFNO(width=0)

        with self.assertRaisesRegex(ValueError, "layers must be positive"):
            MarineFNO(layers=0)

        with self.assertRaisesRegex(ValueError, "Fourier mode counts must be positive"):
            MarineFNO(modes_x=0)

    def test_forward_rejects_invalid_input_dimensions(self):
        with self.assertRaisesRegex(ValueError, "Inputs must have shape"):
            self.model(torch.zeros(1, 3, 8, 10, 12))

    def test_forward_rejects_invalid_channel_count(self):
        inputs = torch.zeros(1, INPUT_CHANNELS - 1, 8, 10)

        with self.assertRaisesRegex(ValueError, "Expected 6 input channels"):
            self.model(inputs)

    def test_load_checkpoint_rejects_incompatible_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "invalid.pt"
            torch.save({"contract_version": "incompatible"}, checkpoint_path)

            with self.assertRaisesRegex(ValueError, "contract version"):
                load_checkpoint(checkpoint_path)

    def test_load_checkpoint_rejects_missing_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "invalid.pt"
            torch.save({"contract_version": CONTRACT_VERSION}, checkpoint_path)

            with self.assertRaisesRegex(ValueError, "architecture metadata"):
                load_checkpoint(checkpoint_path)

    def test_load_checkpoint_rejects_incompatible_forecast_horizon(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "invalid.pt"
            torch.save(
                {
                    "contract_version": CONTRACT_VERSION,
                    "architecture": {"forecast_steps": 3},
                },
                checkpoint_path,
            )

            with self.assertRaisesRegex(ValueError, "forecast horizon"):
                load_checkpoint(checkpoint_path)

    def test_spectral_conv_rejects_zero_modes(self):
        with self.assertRaisesRegex(ValueError, "Fourier mode counts"):
            SpectralConv2d(
                in_channels=1,
                out_channels=1,
                modes_x=0,
                modes_y=1,
            )

    def test_spectral_conv_handles_grid_smaller_than_modes(self):
        layer = SpectralConv2d(
            in_channels=1,
            out_channels=1,
            modes_x=1,
            modes_y=1,
        )
        inputs = torch.randn(1, 1, 1, 4)

        outputs = layer(inputs)

        self.assertEqual(outputs.shape, inputs.shape)

    def test_predict_applies_input_normalization(self):
        input_mean = torch.zeros((1, INPUT_CHANNELS, 1, 1))
        input_std = torch.ones((1, INPUT_CHANNELS, 1, 1))

        input_mean[:, 0] = 20.0
        input_std[:, 0] = 10.0

        raw_result = predict(self.request, self.model)

        normalized_result = predict(
            self.request,
            self.model,
            input_mean=input_mean,
            input_std=input_std,
        )

        self.assertNotEqual(
            raw_result.water_level_m,
            normalized_result.water_level_m,
        )

    def test_checkpoint_without_input_statistics_remains_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "legacy.pt"

            save_checkpoint(self.model, checkpoint_path, epoch=1)

            restored_model, checkpoint = load_checkpoint(checkpoint_path)

            self.assertIsInstance(restored_model, MarineFNO)
            self.assertNotIn("input_mean", checkpoint)
            self.assertNotIn("input_std", checkpoint)
