"""Tests for marine FNO v2 evaluation utilities."""

import tempfile
from pathlib import Path

import numpy as np
import torch
from django.test import SimpleTestCase

from port_twin_app.services.marine_ai.evaluation_v2 import (
    RegressionMetrics,
    _harmonic_tide_prediction,
    _metrics,
    _persistence_prediction,
    evaluate_checkpoint_v2,
    evaluate_model_v2,
)
from port_twin_app.services.marine_ai.model_v2 import (
    MarineFNOV2,
    save_checkpoint,
)
from port_twin_app.services.marine_ai.preparation_v2 import (
    MarineFNOV2Sample,
)


def make_sample(name="sample", offset=0.0):
    ny, nx = 5, 5
    mask = np.ones((ny, nx), dtype=bool)
    mask[0, 0] = False

    initial = np.zeros((3, ny, nx), dtype=np.float32)
    initial[0] = offset
    initial[1] = 0.1
    initial[2] = 0.05

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
        bathymetry_m=np.full((ny, nx), 20.0, dtype=np.float32),
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


class MarineFNOV2EvaluationTests(SimpleTestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.sample = make_sample()
        self.model = MarineFNOV2(
            width=4,
            modes_x=2,
            modes_y=2,
            layers=1,
        )

        self.input_mean = torch.zeros((1, 13, 1, 1))
        self.input_std = torch.ones((1, 13, 1, 1))
        self.target_mean = torch.zeros((1, 1, 3, 1, 1))
        self.target_std = torch.ones((1, 1, 3, 1, 1))

    def test_persistence_repeats_initial_state(self):
        prediction = _persistence_prediction(self.sample.initial_state)

        self.assertEqual(prediction.shape, (6, 3, 5, 5))

        for step in prediction:
            np.testing.assert_allclose(step, self.sample.initial_state)

    def test_harmonic_tide_changes_water_level_and_persists_currents(self):
        prediction = _harmonic_tide_prediction(self.sample)

        self.assertEqual(prediction.shape, (6, 3, 5, 5))
        self.assertFalse(np.allclose(prediction[:, 0], self.sample.initial_state[0]))
        np.testing.assert_allclose(
            prediction[:, 1],
            np.broadcast_to(self.sample.initial_state[1], (6, 5, 5)),
        )
        np.testing.assert_allclose(
            prediction[:, 2],
            np.broadcast_to(self.sample.initial_state[2], (6, 5, 5)),
        )

    def test_metrics_ignore_land_cells(self):
        mask = self.sample.water_mask
        prediction = np.zeros((6, 3, 5, 5), dtype=np.float32)
        target = np.ones_like(prediction)
        target[:, :, ~mask] = np.nan
        prediction[:, :, ~mask] = 9999.0

        metrics = _metrics([prediction], [target], [mask])

        for result in metrics.values():
            self.assertEqual(result, RegressionMetrics(mae=1.0, rmse=1.0))

    def test_evaluation_returns_all_three_comparisons(self):
        result = evaluate_model_v2(
            self.model,
            (self.sample,),
            self.input_mean,
            self.input_std,
            self.target_mean,
            self.target_std,
        )

        self.assertEqual(result.samples_evaluated, 1)
        self.assertIn("water_level_m", result.fno_metrics)
        self.assertIn("current_u_ms", result.persistence_metrics)
        self.assertIn("current_v_ms", result.harmonic_tide_metrics)
        self.assertEqual(result.out_of_domain_models, ("fno-0.4.0",))

    def test_evaluation_rejects_empty_samples(self):
        with self.assertRaises(ValueError):
            evaluate_model_v2(
                self.model,
                (),
                self.input_mean,
                self.input_std,
                self.target_mean,
                self.target_std,
            )

    def test_evaluation_rejects_invalid_normalization_shape(self):
        with self.assertRaises(ValueError):
            evaluate_model_v2(
                self.model,
                (self.sample,),
                torch.zeros((1, 6, 1, 1)),
                self.input_std,
                self.target_mean,
                self.target_std,
            )

    def test_checkpoint_evaluation_uses_saved_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "marine_fno_v2.pt"

            save_checkpoint(
                self.model,
                checkpoint_path,
                input_mean=self.input_mean,
                input_std=self.input_std,
                target_mean=self.target_mean,
                target_std=self.target_std,
            )

            result = evaluate_checkpoint_v2(
                checkpoint_path,
                (self.sample,),
            )

            self.assertEqual(result.samples_evaluated, 1)
            self.assertEqual(result.model_version, "fno-0.5.0")
