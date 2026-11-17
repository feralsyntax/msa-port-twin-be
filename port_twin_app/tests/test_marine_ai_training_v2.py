import tempfile
from pathlib import Path

import numpy as np
import torch
from django.test import SimpleTestCase

from port_twin_app.services.marine_ai.model_v2 import (
    INPUT_CHANNELS,
    MarineFNOV2,
    load_checkpoint,
)
from port_twin_app.services.marine_ai.preparation_v2 import (
    MarineFNOV2Sample,
    MarineFNOV2Split,
)
from port_twin_app.services.marine_ai.training_v2 import (
    TrainingV2Config,
    _continuity_loss,
    _evaluate,
    _fit_input_normalization,
    _fit_target_normalization,
    _masked_supervised_loss,
    build_input_tensor,
    train_model_v2,
)


def make_sample(name="sample", offset=0.0):
    ny, nx = 5, 5
    mask = np.ones((ny, nx), dtype=bool)
    mask[0, 0] = False

    initial = np.zeros((3, ny, nx), dtype=np.float32)
    initial[0] = offset
    initial[1] = 0.1
    initial[2] = 0.05

    bathymetry = np.full((ny, nx), 20.0, dtype=np.float32)

    target = np.zeros((6, 3, ny, nx), dtype=np.float32)
    target[:, 0] = offset + 0.01
    target[:, 1] = 0.1
    target[:, 2] = 0.05
    target[:, :, ~mask] = np.nan

    return MarineFNOV2Sample(
        sample_id=name,
        scenario_id=name,
        start_time_seconds=0.0,
        initial_state=initial,
        bathymetry_m=bathymetry,
        water_mask=mask,
        easting_m=np.arange(nx, dtype=np.float64) * 250.0,
        northing_m=np.arange(ny, dtype=np.float64) * 250.0,
        target=target,
        forcing={
            "wind_u_ms": 1.0,
            "wind_v_ms": 0.0,
            "tide_amplitude_m": 0.5,
            "tide_phase_radians": 0.25,
            "tide_period_seconds": 44712.0,
            "tidal_boundary": "east",
        },
    )


class MarineFNOV2TrainingTests(SimpleTestCase):
    def test_build_input_tensor_uses_thirteen_channels(self):
        sample = make_sample()
        inputs = build_input_tensor(sample)

        self.assertEqual(tuple(inputs.shape), (1, INPUT_CHANNELS, 5, 5))
        self.assertTrue(torch.isfinite(inputs).all())

        self.assertEqual(inputs[0, 4, 1, 1].item(), 1.0)
        self.assertEqual(inputs[0, 7, 1, 1].item(), 1.0)
        self.assertAlmostEqual(
            inputs[0, 10, 1, 1].item(),
            np.sin(0.25),
            places=6,
        )
        self.assertAlmostEqual(
            inputs[0, 11, 1, 1].item(),
            np.cos(0.25),
            places=6,
        )

        # The final column identifies the eastern boundary.
        self.assertTrue(torch.all(inputs[0, 12, :, -1] == 1.0))
        self.assertTrue(torch.all(inputs[0, 12, :, :-1] == 0.0))

    def test_input_normalization_uses_training_samples(self):
        samples = (make_sample("one", 1.0), make_sample("two", 3.0))

        mean, std = _fit_input_normalization(samples, "cpu")

        self.assertEqual(tuple(mean.shape), (1, 13, 1, 1))
        self.assertEqual(tuple(std.shape), (1, 13, 1, 1))
        self.assertTrue(torch.isfinite(mean).all())
        self.assertTrue(torch.all(std > 0))

    def test_target_normalization_excludes_land_nan_values(self):
        samples = (make_sample("one", 1.0), make_sample("two", 3.0))

        mean, std = _fit_target_normalization(samples, "cpu")

        self.assertEqual(tuple(mean.shape), (1, 1, 3, 1, 1))
        self.assertEqual(tuple(std.shape), (1, 1, 3, 1, 1))
        self.assertTrue(torch.isfinite(mean).all())
        self.assertTrue(torch.isfinite(std).all())
        self.assertTrue(torch.all(std > 0))

    def test_masked_supervised_loss_ignores_invalid_land_values(self):
        predictions = torch.zeros(1, 6, 3, 5, 5)
        targets = torch.ones_like(predictions)
        mask = torch.ones(1, 5, 5, dtype=torch.bool)
        mask[:, 0, 0] = False

        targets[:, :, :, 0, 0] = float("nan")

        loss = _masked_supervised_loss(predictions, targets, mask)

        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(loss.item(), 1.0)

    def test_continuity_loss_is_finite_and_differentiable(self):
        sample = make_sample()

        predictions = torch.zeros(
            1,
            6,
            3,
            5,
            5,
            requires_grad=True,
        )
        initial = torch.as_tensor(sample.initial_state)
        bathymetry = torch.as_tensor(sample.bathymetry_m)
        mask = torch.as_tensor(sample.water_mask)
        easting = torch.as_tensor(sample.easting_m, dtype=torch.float32)
        northing = torch.as_tensor(sample.northing_m, dtype=torch.float32)

        loss = _continuity_loss(
            predictions,
            initial,
            bathymetry,
            mask,
            easting,
            northing,
        )

        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(predictions.grad)
        self.assertTrue(torch.isfinite(predictions.grad).all())

    def test_training_saves_loadable_best_checkpoint(self):
        train = (
            make_sample("train-one", 0.0),
            make_sample("train-two", 0.2),
        )
        validation = (make_sample("validation", 0.1),)
        test = (make_sample("test", 0.3),)

        dataset = MarineFNOV2Split(
            train=train,
            validation=validation,
            test=test,
            scenario_partitions={
                "train-one": "train",
                "train-two": "train",
                "validation": "validation",
                "test": "test",
            },
        )

        model = MarineFNOV2(
            width=4,
            modes_x=2,
            modes_y=2,
            layers=1,
        )

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "marine_fno_v2.pt"

            result = train_model_v2(
                model,
                dataset,
                TrainingV2Config(
                    epochs=1,
                    learning_rate=1e-3,
                    checkpoint_path=str(checkpoint_path),
                ),
            )

            self.assertEqual(result.model_version, "fno-0.5.0")
            self.assertEqual(result.epochs_completed, 1)
            self.assertEqual(result.best_epoch, 1)
            self.assertTrue(checkpoint_path.is_file())

            history_fields = (
                "training_supervised_losses",
                "training_continuity_losses",
                "validation_supervised_losses",
                "validation_continuity_losses",
            )

            for field in history_fields:
                history = getattr(result, field)

                self.assertEqual(len(history), result.epochs_completed)
                self.assertTrue(np.isfinite(history).all())
                self.assertTrue(all(value >= 0 for value in history))

            loaded_model, checkpoint = load_checkpoint(checkpoint_path)

            self.assertIsInstance(loaded_model, MarineFNOV2)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertIn("training", checkpoint)

            for field in history_fields:
                self.assertIn(field, checkpoint["training"])
                self.assertEqual(
                    len(checkpoint["training"][field]),
                    result.epochs_completed,
                )
                self.assertTrue(np.isfinite(checkpoint["training"][field]).all())
                self.assertTrue(
                    all(value >= 0 for value in checkpoint["training"][field])
                )

            self.assertIn("input_mean", checkpoint)
            self.assertIn("target_mean", checkpoint)

    def test_training_config_rejects_invalid_physics_weight(self):
        with self.assertRaisesRegex(ValueError, "physics_loss_weight"):
            TrainingV2Config(physics_loss_weight=-0.1)

    def test_training_config_rejects_invalid_continuity_scale(self):
        with self.assertRaisesRegex(ValueError, "continuity_residual_scale_ms"):
            TrainingV2Config(continuity_residual_scale_ms=0.0)

    def test_evaluate_returns_separate_loss_components(self):
        samples = (make_sample("train-one", 0.0),)
        validation = (make_sample("validation", 0.1),)

        model = MarineFNOV2(
            width=4,
            modes_x=2,
            modes_y=2,
            layers=1,
        )

        input_mean, input_std = _fit_input_normalization(samples, "cpu")
        target_mean, target_std = _fit_target_normalization(samples, "cpu")

        total, supervised, continuity = _evaluate(
            model,
            validation,
            "cpu",
            input_mean,
            input_std,
            target_mean,
            target_std,
            TrainingV2Config(epochs=1),
        )

        self.assertTrue(np.isfinite([total, supervised, continuity]).all())
        self.assertTrue(all(value >= 0 for value in (total, supervised, continuity)))
        self.assertAlmostEqual(
            total,
            supervised + 0.1 * continuity,
            places=5,
        )
