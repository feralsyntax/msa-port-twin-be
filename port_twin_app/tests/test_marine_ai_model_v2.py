import tempfile
from pathlib import Path

import torch
from django.test import SimpleTestCase

from port_twin_app.services.marine_ai.model_v2 import (
    FORECAST_STEPS,
    INPUT_CHANNELS,
    INPUT_STATISTICS_SHAPE,
    OUTPUT_CHANNELS,
    TARGET_STATISTICS_SHAPE,
    MarineFNOV2,
    load_checkpoint,
    save_checkpoint,
)


class MarineFNOV2ModelTests(SimpleTestCase):
    def test_model_returns_expected_output_shape(self):
        model = MarineFNOV2(
            width=8,
            modes_x=4,
            modes_y=4,
            layers=2,
        )

        inputs = torch.randn(2, INPUT_CHANNELS, 8, 10)
        outputs = model(inputs)

        self.assertEqual(
            tuple(outputs.shape),
            (2, FORECAST_STEPS, OUTPUT_CHANNELS, 8, 10),
        )

    def test_model_rejects_wrong_input_dimensions(self):
        model = MarineFNOV2(width=8, modes_x=4, modes_y=4, layers=2)

        with self.assertRaisesRegex(ValueError, "shape"):
            model(torch.randn(2, INPUT_CHANNELS, 8))

    def test_model_rejects_wrong_channel_count(self):
        model = MarineFNOV2(width=8, modes_x=4, modes_y=4, layers=2)

        with self.assertRaisesRegex(ValueError, "Expected 13 input channels"):
            model(torch.randn(2, 6, 8, 8))

    def test_model_rejects_non_finite_inputs(self):
        model = MarineFNOV2(width=8, modes_x=4, modes_y=4, layers=2)

        inputs = torch.randn(1, INPUT_CHANNELS, 8, 8)
        inputs[0, 0, 0, 0] = float("nan")

        with self.assertRaisesRegex(ValueError, "non-finite"):
            model(inputs)

    def test_model_rejects_integer_inputs(self):
        model = MarineFNOV2(width=8, modes_x=4, modes_y=4, layers=2)

        inputs = torch.ones(1, INPUT_CHANNELS, 8, 8, dtype=torch.int64)

        with self.assertRaisesRegex(ValueError, "floating-point"):
            model(inputs)

    def test_rejects_invalid_architecture(self):
        with self.assertRaisesRegex(ValueError, "width"):
            MarineFNOV2(width=0)

        with self.assertRaisesRegex(ValueError, "layers"):
            MarineFNOV2(layers=0)

        with self.assertRaisesRegex(ValueError, "Fourier"):
            MarineFNOV2(modes_x=0)

    def test_checkpoint_round_trip(self):
        model = MarineFNOV2(
            width=8,
            modes_x=4,
            modes_y=4,
            layers=2,
        )

        input_mean = torch.zeros(INPUT_STATISTICS_SHAPE)
        input_std = torch.ones(INPUT_STATISTICS_SHAPE)

        target_mean = torch.zeros(TARGET_STATISTICS_SHAPE)
        target_std = torch.ones(TARGET_STATISTICS_SHAPE)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "marine_fno_v2.pt"

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

            self.assertEqual(checkpoint["model_version"], "fno-0.5.0")
            self.assertEqual(checkpoint["dataset_version"], "v2")
            self.assertEqual(checkpoint["input_channels"], 13)
            self.assertEqual(checkpoint["epoch"], 3)

            self.assertEqual(
                checkpoint["input_mean"].shape,
                INPUT_STATISTICS_SHAPE,
            )
            self.assertEqual(
                checkpoint["target_mean"].shape,
                TARGET_STATISTICS_SHAPE,
            )

            original_state = model.state_dict()
            loaded_state = loaded_model.state_dict()

            self.assertEqual(original_state.keys(), loaded_state.keys())

            for key in original_state:
                self.assertTrue(
                    torch.equal(original_state[key], loaded_state[key]),
                    msg=f"Checkpoint weight mismatch: {key}",
                )

            self.assertFalse(loaded_model.training)

    def test_checkpoint_rejects_wrong_dataset_version(self):
        model = MarineFNOV2(width=8, modes_x=4, modes_y=4, layers=2)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "marine_fno_v2.pt"
            save_checkpoint(model, path)

            checkpoint = torch.load(path, weights_only=True)
            checkpoint["dataset_version"] = "v1"
            torch.save(checkpoint, path)

            with self.assertRaisesRegex(ValueError, "dataset_version"):
                load_checkpoint(path)

    def test_checkpoint_rejects_wrong_input_contract(self):
        model = MarineFNOV2(width=8, modes_x=4, modes_y=4, layers=2)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "marine_fno_v2.pt"
            save_checkpoint(model, path)

            checkpoint = torch.load(path, weights_only=True)
            checkpoint["input_channels"] = 6
            torch.save(checkpoint, path)

            with self.assertRaisesRegex(ValueError, "input_channels"):
                load_checkpoint(path)

    def test_checkpoint_rejects_incomplete_normalization_statistics(self):
        model = MarineFNOV2(width=8, modes_x=4, modes_y=4, layers=2)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "marine_fno_v2.pt"
            save_checkpoint(model, path)

            checkpoint = torch.load(path, weights_only=True)
            checkpoint["input_mean"] = torch.zeros(INPUT_STATISTICS_SHAPE)
            torch.save(checkpoint, path)

            with self.assertRaisesRegex(ValueError, "both input_mean"):
                load_checkpoint(path)

    def test_save_rejects_incomplete_normalization_statistics(self):
        model = MarineFNOV2(width=8, modes_x=4, modes_y=4, layers=2)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "marine_fno_v2.pt"

            with self.assertRaisesRegex(ValueError, "both be supplied"):
                save_checkpoint(
                    model,
                    path,
                    input_mean=torch.zeros(INPUT_STATISTICS_SHAPE),
                )
