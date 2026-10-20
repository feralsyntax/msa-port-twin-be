"""Tests for marine FNO training utilities."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import torch
from django.test import SimpleTestCase

from port_twin_app.services.marine_ai.contracts import (
    MarineForecastConfig,
    MarineGrid,
    MarinePredictionRequest,
    MarinePredictionResult,
)
from port_twin_app.services.marine_ai.dataset import (
    MarineDatasetSample,
    split_dataset,
)
from port_twin_app.services.marine_ai.model import MarineFNO
from port_twin_app.services.marine_ai.training import (
    TrainingConfig,
    _target_tensor,
    train_model,
)


def make_sample(sample_id, start_time, offset=0.0):
    steps, ny, nx = 2, 8, 8

    request = MarinePredictionRequest(
        grid=MarineGrid(
            west=39.5,
            south=-4.2,
            east=39.8,
            north=-3.9,
            nx=nx,
            ny=ny,
        ),
        forecast=MarineForecastConfig(
            start_time=start_time.isoformat(),
            steps=steps,
            step_seconds=3600,
        ),
        bathymetry_m=[[20.0 for _ in range(nx)] for _ in range(ny)],
        initial_water_level_m=[[0.5 + offset for _ in range(nx)] for _ in range(ny)],
        wind_u_ms=[[[2.0 for _ in range(nx)] for _ in range(ny)] for _ in range(steps)],
        wind_v_ms=[[[1.0 for _ in range(nx)] for _ in range(ny)] for _ in range(steps)],
        boundary_conditions={
            "west": {"water_level_m": [0.1] * steps},
        },
    )

    target = MarinePredictionResult(
        water_level_m=[
            [[0.1 + offset for _ in range(nx)] for _ in range(ny)] for _ in range(steps)
        ],
        current_u_ms=[
            [[0.2 for _ in range(nx)] for _ in range(ny)] for _ in range(steps)
        ],
        current_v_ms=[
            [[0.3 for _ in range(nx)] for _ in range(ny)] for _ in range(steps)
        ],
        model_version="fixture-target-v1",
    )

    return MarineDatasetSample(
        sample_id=sample_id,
        request=request,
        target=target,
        source="unit-test-fixture",
        source_version="1",
        license_name="test-only",
    )


class MarineTrainingTests(SimpleTestCase):
    def setUp(self):
        torch.manual_seed(7)
        start = datetime.fromisoformat("2026-01-01T00:00:00+00:00")

        # Wide gaps ensure the forecast windows do not cross split boundaries.
        samples = [
            make_sample(
                f"sample-{index}",
                start + timedelta(days=index * 2),
                offset=index * 0.05,
            )
            for index in range(20)
        ]
        self.dataset = split_dataset(samples)
        self.model = MarineFNO(
            width=4,
            modes_x=2,
            modes_y=2,
            layers=1,
        )

    def test_target_tensor_has_expected_channel_order_and_shape(self):
        sample = self.dataset.train[0]
        target = _target_tensor(sample.target, "cpu")

        self.assertEqual(tuple(target.shape), (1, 2, 3, 8, 8))
        self.assertAlmostEqual(
            target[0, 0, 0, 0, 0].item(),
            sample.target.water_level_m[0][0][0],
        )
        self.assertAlmostEqual(
            target[0, 0, 1, 0, 0].item(),
            sample.target.current_u_ms[0][0][0],
        )
        self.assertAlmostEqual(
            target[0, 0, 2, 0, 0].item(),
            sample.target.current_v_ms[0][0][0],
        )

    def test_training_updates_model_and_saves_best_checkpoint(self):
        before = [parameter.detach().clone() for parameter in self.model.parameters()]

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "marine_fno.pt"

            result = train_model(
                self.model,
                self.dataset,
                TrainingConfig(
                    epochs=2,
                    learning_rate=1e-3,
                    device="cpu",
                    checkpoint_path=str(checkpoint_path),
                ),
            )

            self.assertTrue(checkpoint_path.exists())
            self.assertEqual(result.epochs_completed, 2)
            self.assertEqual(len(result.training_losses), 2)
            self.assertEqual(len(result.validation_losses), 2)
            self.assertTrue(
                all(
                    torch.isfinite(torch.tensor(loss))
                    for loss in result.training_losses
                )
            )
            self.assertTrue(
                all(
                    torch.isfinite(torch.tensor(loss))
                    for loss in result.validation_losses
                )
            )
            self.assertEqual(
                result.best_validation_loss,
                min(result.validation_losses),
            )

            from port_twin_app.services.marine_ai.model import load_checkpoint

            restored_model, checkpoint = load_checkpoint(checkpoint_path)
            self.assertEqual(
                checkpoint["epoch"],
                result.validation_losses.index(result.best_validation_loss) + 1,
            )
            self.assertIsInstance(restored_model, MarineFNO)

        self.assertTrue(
            any(
                not torch.equal(old, new.detach())
                for old, new in zip(before, self.model.parameters())
            )
        )

    def test_training_rejects_wrong_model_type(self):
        with self.assertRaises(TypeError):
            train_model(
                torch.nn.Linear(1, 1),
                self.dataset,
                TrainingConfig(epochs=1),
            )

    def test_training_config_rejects_invalid_epochs(self):
        with self.assertRaises(ValueError):
            TrainingConfig(epochs=0)
