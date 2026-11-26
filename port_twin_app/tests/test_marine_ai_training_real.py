import tempfile
from pathlib import Path

import numpy as np
import torch
from django.test import SimpleTestCase

from port_twin_app.services.marine_ai.model_real import MarineFNOReal
from port_twin_app.services.marine_ai.preparation_real import (
    MarineFNORealSample,
    MarineFNORealSplit,
)
from port_twin_app.services.marine_ai.training_real import (
    TrainingRealConfig,
    _fit_input_normalization,
    _fit_target_normalization,
    _stack_samples,
    masked_mse_loss,
    train_model_real,
)


def make_sample(sample_id: str, offset: float = 0.0):
    """Create a small prepared real-data sample for training tests."""

    height = 8
    width = 8

    mask = np.ones((height, width), dtype=bool)
    mask[-1, -1] = False

    inputs = np.zeros((6, height, width), dtype=np.float32)

    for channel in range(3):
        inputs[channel, mask] = offset + channel + 1.0

    inputs[3] = mask.astype(np.float32)

    longitude = np.linspace(-1.0, 1.0, width)
    latitude = np.linspace(-1.0, 1.0, height)
    longitude_grid, latitude_grid = np.meshgrid(longitude, latitude)

    inputs[4] = longitude_grid
    inputs[5] = latitude_grid

    target = np.ones((6, 3, height, width), dtype=np.float32) * (offset + 1.0)
    target[:, :, ~mask] = np.nan

    return MarineFNORealSample(
        sample_id=sample_id,
        start_time=np.datetime64("2020-01-01"),
        inputs=inputs,
        target=target,
    )


class MarineFNORealTrainingTests(SimpleTestCase):
    def test_stack_samples_replaces_land_targets_with_zero(self):
        samples = (make_sample("sample-1"),)

        inputs, targets, masks = _stack_samples(samples)

        self.assertEqual(inputs.shape, (1, 6, 8, 8))
        self.assertEqual(targets.shape, (1, 6, 3, 8, 8))
        self.assertEqual(masks.shape, (1, 8, 8))
        self.assertEqual(targets[0, 0, 0, -1, -1].item(), 0.0)
        self.assertFalse(masks[0, -1, -1].item())

    def test_input_normalization_uses_water_cells(self):
        inputs, _, masks = _stack_samples(
            (make_sample("sample-1"), make_sample("sample-2", 2.0))
        )

        mean, std = _fit_input_normalization(inputs, masks)

        self.assertEqual(tuple(mean.shape), (1, 6, 1, 1))
        self.assertEqual(tuple(std.shape), (1, 6, 1, 1))
        self.assertTrue(torch.isfinite(mean).all())
        self.assertTrue(torch.isfinite(std).all())
        self.assertTrue(torch.all(std > 0))

    def test_target_normalization_uses_water_cells(self):
        _, targets, masks = _stack_samples(
            (make_sample("sample-1"), make_sample("sample-2", 2.0))
        )

        mean, std = _fit_target_normalization(targets, masks)

        self.assertEqual(tuple(mean.shape), (1, 1, 3, 1, 1))
        self.assertEqual(tuple(std.shape), (1, 1, 3, 1, 1))
        self.assertTrue(torch.isfinite(mean).all())
        self.assertTrue(torch.isfinite(std).all())
        self.assertTrue(torch.all(std > 0))

    def test_masked_loss_ignores_land_cells(self):
        predictions = torch.zeros(1, 6, 3, 2, 2)
        targets = torch.ones_like(predictions)
        masks = torch.tensor([[[True, True], [True, False]]])

        # Give the land cell a very large error.
        predictions[:, :, :, 1, 1] = 1000.0

        loss = masked_mse_loss(predictions, targets, masks)

        self.assertAlmostEqual(loss.item(), 1.0, places=6)

    def test_training_saves_best_checkpoint(self):
        train_samples = (
            make_sample("train-1"),
            make_sample("train-2", 0.2),
        )
        validation_samples = (make_sample("validation-1", 0.1),)
        test_samples = (make_sample("test-1", 0.3),)

        dataset = MarineFNORealSplit(
            train=train_samples,
            validation=validation_samples,
            test=test_samples,
            longitude=np.linspace(40.0, 41.0, 8),
            latitude=np.linspace(-5.0, -4.0, 8),
            water_mask=np.ones((8, 8), dtype=bool),
        )

        # Keep this a small, fast CPU test.
        model = MarineFNOReal(
            width=4,
            modes_x=2,
            modes_y=2,
            layers=1,
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "real.pt"

            result = train_model_real(
                model,
                dataset,
                TrainingRealConfig(
                    epochs=1,
                    batch_size=2,
                    learning_rate=1e-3,
                    device="cpu",
                    checkpoint_path=str(checkpoint_path),
                ),
            )

            self.assertTrue(checkpoint_path.is_file())
            self.assertEqual(result.epochs_completed, 1)
            self.assertEqual(result.best_epoch, 1)
            self.assertEqual(len(result.training_losses), 1)
            self.assertEqual(len(result.validation_losses), 1)
            self.assertTrue(np.isfinite(result.best_validation_loss))
