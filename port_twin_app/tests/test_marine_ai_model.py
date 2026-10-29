"""Tests for the marine FNO model and inference utilities."""

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
    CONTRACT_VERSION,
    INPUT_CHANNELS,
    MODEL_VERSION,
    MarineFNO,
    load_checkpoint,
    predict,
    predict_from_checkpoint,
    request_to_tensor,
    save_checkpoint,
)


def make_request(steps=3, ny=8, nx=10):
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
            steps=steps,
            step_seconds=3600,
        ),
        bathymetry_m=[[20.0 for _ in range(nx)] for _ in range(ny)],
        initial_water_level_m=[[0.5 for _ in range(nx)] for _ in range(ny)],
        wind_u_ms=[[[2.0 for _ in range(nx)] for _ in range(ny)] for _ in range(steps)],
        wind_v_ms=[[[1.0 for _ in range(nx)] for _ in range(ny)] for _ in range(steps)],
        boundary_conditions={
            "west": {
                "water_level_m": [0.1] * steps,
            },
            "north": {
                "normal_velocity_ms": [0.2] * steps,
            },
        },
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

    def test_request_to_tensor_has_expected_shape(self):
        inputs = request_to_tensor(self.request)

        self.assertEqual(
            tuple(inputs.shape),
            (1, 3, INPUT_CHANNELS, 8, 10),
        )
        self.assertTrue(torch.isfinite(inputs).all())

    def test_boundary_conditions_are_encoded_in_expected_channels(self):
        inputs = request_to_tensor(self.request)

        # Channel 7: west water-level boundary.
        self.assertTrue(
            torch.allclose(
                inputs[0, :, 7, :, 0],
                torch.full((3, 8), 0.1),
            )
        )

        # Channel 14: north normal-velocity boundary.
        self.assertTrue(
            torch.allclose(
                inputs[0, :, 14, -1, :],
                torch.full((3, 10), 0.2),
            )
        )

        # Boundary values should not fill unrelated interior locations.
        self.assertTrue(torch.all(inputs[0, :, 7, :, 1:] == 0))
        self.assertTrue(torch.all(inputs[0, :, 14, :-1, :] == 0))

    def test_forward_returns_expected_shape(self):
        inputs = request_to_tensor(self.request)
        outputs = self.model(inputs)

        self.assertEqual(
            tuple(outputs.shape),
            (1, 3, 3, 8, 10),
        )
        self.assertTrue(torch.isfinite(outputs).all())

    def test_optimizer_updates_model_parameters(self):
        inputs = request_to_tensor(self.request)
        before = [parameter.detach().clone() for parameter in self.model.parameters()]

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3,
        )

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
        self.assertEqual(len(result.water_level_m), 3)
        self.assertEqual(len(result.current_u_ms), 3)
        self.assertEqual(len(result.current_v_ms), 3)
        self.assertEqual(len(result.water_level_m[0]), 8)
        self.assertEqual(len(result.water_level_m[0][0]), 10)

    def test_predict_reverses_target_normalization(self):
        """Normalized predictions must be converted back to physical units."""

        mean = torch.tensor(
            [0.5, 2.0, -1.0],
            dtype=torch.float32,
        ).view(1, 1, 3, 1, 1)

        std = torch.tensor(
            [0.2, 0.4, 0.6],
            dtype=torch.float32,
        ).view(1, 1, 3, 1, 1)

        # Force the model to produce known normalized outputs of zero.
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.zero_()

        result = predict(
            self.request,
            self.model,
            target_mean=mean,
            target_std=std,
        )

        # A normalized output of zero becomes the corresponding target mean.
        self.assertAlmostEqual(
            result.water_level_m[0][0][0],
            0.5,
            places=5,
        )
        self.assertAlmostEqual(
            result.current_u_ms[0][0][0],
            2.0,
            places=5,
        )
        self.assertAlmostEqual(
            result.current_v_ms[0][0][0],
            -1.0,
            places=5,
        )

    def test_predict_requires_both_normalization_statistics(self):
        mean = torch.zeros((1, 1, 3, 1, 1))

        with self.assertRaises(ValueError):
            predict(
                self.request,
                self.model,
                target_mean=mean,
            )

        with self.assertRaises(ValueError):
            predict(
                self.request,
                self.model,
                target_std=torch.ones_like(mean),
            )

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

            save_checkpoint(
                self.model,
                checkpoint_path,
                epoch=3,
            )

            restored_model, checkpoint = load_checkpoint(
                checkpoint_path,
            )

            self.assertIsInstance(restored_model, MarineFNO)
            self.assertEqual(checkpoint["epoch"], 3)
            self.assertEqual(
                checkpoint["model_version"],
                MODEL_VERSION,
            )
            self.assertEqual(
                checkpoint["architecture"]["width"],
                4,
            )

            original_output = self.model(request_to_tensor(self.request))
            restored_output = restored_model(request_to_tensor(self.request))

            self.assertTrue(
                torch.allclose(
                    original_output,
                    restored_output,
                    atol=1e-6,
                )
            )

    def test_predict_from_checkpoint_uses_saved_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "marine_fno.pt"

            save_checkpoint(
                self.model,
                checkpoint_path,
                epoch=2,
            )

            # Simulate the normalization metadata saved by training.py.
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )

            checkpoint["target_mean"] = torch.tensor(
                [0.5, 2.0, -1.0],
                dtype=torch.float32,
            ).view(1, 1, 3, 1, 1)

            checkpoint["target_std"] = torch.tensor(
                [0.2, 0.4, 0.6],
                dtype=torch.float32,
            ).view(1, 1, 3, 1, 1)

            torch.save(checkpoint, checkpoint_path)

            # The zeroed model produces zero normalized outputs, so inference
            # should return the checkpoint's means in physical units.
            with torch.no_grad():
                for parameter in self.model.parameters():
                    parameter.zero_()

            save_checkpoint(
                self.model,
                checkpoint_path,
                epoch=2,
            )

            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            checkpoint["target_mean"] = torch.tensor(
                [0.5, 2.0, -1.0],
                dtype=torch.float32,
            ).view(1, 1, 3, 1, 1)
            checkpoint["target_std"] = torch.tensor(
                [0.2, 0.4, 0.6],
                dtype=torch.float32,
            ).view(1, 1, 3, 1, 1)
            torch.save(checkpoint, checkpoint_path)

            result = predict_from_checkpoint(
                self.request,
                checkpoint_path,
            )

            self.assertAlmostEqual(
                result.water_level_m[0][0][0],
                0.5,
                places=5,
            )
            self.assertAlmostEqual(
                result.current_u_ms[0][0][0],
                2.0,
                places=5,
            )
            self.assertAlmostEqual(
                result.current_v_ms[0][0][0],
                -1.0,
                places=5,
            )

    def test_model_rejects_invalid_architecture_parameters(self):
        with self.assertRaisesRegex(ValueError, "width must be positive"):
            MarineFNO(width=0)

        with self.assertRaisesRegex(ValueError, "layers must be positive"):
            MarineFNO(layers=0)

        with self.assertRaisesRegex(ValueError, "Fourier mode counts must be positive"):
            MarineFNO(modes_x=0)

    def test_forward_rejects_invalid_input_dimensions(self):
        with self.assertRaisesRegex(ValueError, "Inputs must have shape"):
            self.model(torch.zeros(1, 3, 8, 10))

    def test_forward_rejects_invalid_channel_count(self):
        inputs = torch.zeros(1, 3, INPUT_CHANNELS - 1, 8, 10)

        with self.assertRaisesRegex(ValueError, "Expected 15 input channels"):
            self.model(inputs)

    def test_request_to_tensor_rejects_invalid_grid_shape(self):
        request = make_request()
        object.__setattr__(request, "bathymetry_m", [[20.0, 30.0]])

        with self.assertRaisesRegex(ValueError, "bathymetry_m must have shape"):
            request_to_tensor(request)

    def test_request_to_tensor_rejects_non_finite_grid_values(self):
        self.request.bathymetry_m[0][0] = float("nan")

        with self.assertRaisesRegex(ValueError, "bathymetry_m contains non-finite"):
            request_to_tensor(self.request)

    def test_request_to_tensor_rejects_invalid_wind_shape(self):
        request = make_request()
        object.__setattr__(request, "wind_u_ms", [])

        with self.assertRaisesRegex(ValueError, "wind_u_ms must have shape"):
            request_to_tensor(request)

    def test_request_to_tensor_rejects_non_finite_wind_values(self):
        self.request.wind_v_ms[0][0][0] = float("inf")

        with self.assertRaisesRegex(ValueError, "wind_v_ms contains non-finite"):
            request_to_tensor(self.request)

    def test_load_checkpoint_rejects_incompatible_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "invalid.pt"
            torch.save(
                {"contract_version": "incompatible"},
                checkpoint_path,
            )

            with self.assertRaisesRegex(ValueError, "contract version"):
                load_checkpoint(checkpoint_path)

    def test_load_checkpoint_rejects_missing_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "invalid.pt"
            torch.save(
                {"contract_version": CONTRACT_VERSION},
                checkpoint_path,
            )

            with self.assertRaisesRegex(ValueError, "architecture metadata"):
                load_checkpoint(checkpoint_path)
