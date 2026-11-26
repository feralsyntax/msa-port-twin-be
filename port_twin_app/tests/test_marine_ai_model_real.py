import tempfile
from pathlib import Path

import torch
from django.test import SimpleTestCase

from port_twin_app.services.marine_ai.model_real import (
    FORECAST_STEPS,
    INPUT_CHANNELS,
    INPUT_STATISTICS_SHAPE,
    OUTPUT_CHANNELS,
    TARGET_STATISTICS_SHAPE,
    MarineFNOReal,
    load_checkpoint,
    save_checkpoint,
)


class MarineFNORealModelTests(SimpleTestCase):
    def test_forward_returns_expected_forecast_shape(self):
        model = MarineFNOReal(
            width=4,
            modes_x=2,
            modes_y=2,
            layers=1,
        )

        inputs = torch.randn(2, INPUT_CHANNELS, 25, 25)
        predictions = model(inputs)

        self.assertEqual(
            predictions.shape,
            (2, FORECAST_STEPS, OUTPUT_CHANNELS, 25, 25),
        )

    def test_rejects_wrong_input_channel_count(self):
        model = MarineFNOReal(width=4, modes_x=2, modes_y=2, layers=1)

        with self.assertRaisesRegex(ValueError, "Expected 6 input channels"):
            model(torch.randn(1, 5, 25, 25))

    def test_rejects_non_finite_inputs(self):
        model = MarineFNOReal(width=4, modes_x=2, modes_y=2, layers=1)
        inputs = torch.randn(1, INPUT_CHANNELS, 25, 25)
        inputs[0, 0, 0, 0] = float("nan")

        with self.assertRaisesRegex(ValueError, "non-finite"):
            model(inputs)

    def test_checkpoint_round_trip_preserves_model_and_statistics(self):
        model = MarineFNOReal(
            width=4,
            modes_x=2,
            modes_y=2,
            layers=1,
        )

        input_mean = torch.zeros(INPUT_STATISTICS_SHAPE)
        input_std = torch.ones(INPUT_STATISTICS_SHAPE)
        target_mean = torch.zeros(TARGET_STATISTICS_SHAPE)
        target_std = torch.ones(TARGET_STATISTICS_SHAPE)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "real.pt"

            save_checkpoint(
                model,
                path,
                epoch=3,
                input_mean=input_mean,
                input_std=input_std,
                target_mean=target_mean,
                target_std=target_std,
            )

            loaded_model, checkpoint = load_checkpoint(path)

        self.assertEqual(checkpoint["epoch"], 3)
        self.assertEqual(checkpoint["model_version"], "fno-real-0.1.0")

        for name, value in model.state_dict().items():
            torch.testing.assert_close(
                value,
                loaded_model.state_dict()[name],
            )

        self.assertEqual(
            checkpoint["input_mean"].shape,
            INPUT_STATISTICS_SHAPE,
        )
        self.assertEqual(
            checkpoint["target_mean"].shape,
            TARGET_STATISTICS_SHAPE,
        )

    def test_checkpoint_rejects_wrong_model_contract(self):
        model = MarineFNOReal(
            width=4,
            modes_x=2,
            modes_y=2,
            layers=1,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "real.pt"
            save_checkpoint(model, path)

            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
            checkpoint["dataset_version"] = "v2"
            torch.save(checkpoint, path)

            with self.assertRaisesRegex(ValueError, "dataset_version"):
                load_checkpoint(path)

    def test_rejects_invalid_architecture(self):
        with self.assertRaises(ValueError):
            MarineFNOReal(width=0)
