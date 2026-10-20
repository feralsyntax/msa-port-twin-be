"""Tests for marine FNO evaluation utilities."""

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
from port_twin_app.services.marine_ai.evaluation import (
    RegressionMetrics,
    _baseline_prediction,
    _calculate_metrics,
    evaluate_checkpoint,
    evaluate_model,
)
from port_twin_app.services.marine_ai.model import (
    MarineFNO,
    save_checkpoint,
)


def make_sample(sample_id, start_time, offset=0.0):
    """Create a synthetic marine sample for evaluation tests."""

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
        model_version="synthetic-test-target-v1",
    )

    return MarineDatasetSample(
        sample_id=sample_id,
        request=request,
        target=target,
        source="synthetic-test-fixture",
        source_version="1",
        license_name="test-only",
    )


class MarineEvaluationTests(SimpleTestCase):
    def setUp(self):
        torch.manual_seed(7)

        start = datetime.fromisoformat("2026-01-01T00:00:00+00:00")

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

    def test_metrics_calculate_mae_and_rmse(self):
        predictions = [
            torch.tensor(
                [[[[1.0]], [[2.0]], [[3.0]]]],
                dtype=torch.float32,
            )
        ]

        targets = [
            torch.tensor(
                [[[[0.0]], [[0.0]], [[0.0]]]],
                dtype=torch.float32,
            )
        ]

        metrics = _calculate_metrics(predictions, targets)

        self.assertEqual(
            metrics["water_level_m"],
            RegressionMetrics(mae=1.0, rmse=1.0),
        )
        self.assertEqual(
            metrics["current_u_ms"],
            RegressionMetrics(mae=2.0, rmse=2.0),
        )
        self.assertEqual(
            metrics["current_v_ms"],
            RegressionMetrics(mae=3.0, rmse=3.0),
        )

    def test_baseline_persists_initial_water_level_and_zero_currents(self):
        request = self.dataset.test[0].request

        baseline = _baseline_prediction(request)
        baseline.validate_against(request)

        for step in baseline.water_level_m:
            self.assertEqual(
                step,
                request.initial_water_level_m,
            )

        for step in baseline.current_u_ms:
            self.assertTrue(all(value == 0.0 for row in step for value in row))

        for step in baseline.current_v_ms:
            self.assertTrue(all(value == 0.0 for row in step for value in row))

    def test_evaluation_reports_fno_and_baseline_metrics(self):
        result = evaluate_model(
            self.model,
            self.dataset.test,
        )

        self.assertEqual(
            result.samples_evaluated,
            len(self.dataset.test),
        )
        self.assertIn("water_level_m", result.fno_metrics)
        self.assertIn("current_u_ms", result.fno_metrics)
        self.assertIn("current_v_ms", result.fno_metrics)
        self.assertIn("water_level_m", result.baseline_metrics)
        self.assertIn("synthetic-test-fixture", result.data_sources)
        self.assertIn("Synthetic fixtures", result.evaluation_note)

        for metrics in (
            *result.fno_metrics.values(),
            *result.baseline_metrics.values(),
        ):
            self.assertTrue(torch.isfinite(torch.tensor(metrics.mae)))
            self.assertTrue(torch.isfinite(torch.tensor(metrics.rmse)))
            self.assertGreaterEqual(metrics.mae, 0)
            self.assertGreaterEqual(metrics.rmse, 0)

    def test_evaluation_rejects_empty_samples(self):
        with self.assertRaises(ValueError):
            evaluate_model(self.model, ())

    def test_evaluation_requires_both_normalization_statistics(self):
        mean = torch.zeros((1, 1, 3, 1, 1))

        with self.assertRaises(ValueError):
            evaluate_model(
                self.model,
                self.dataset.test,
                target_mean=mean,
            )

    def test_evaluate_checkpoint_uses_saved_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "marine_fno.pt"

            save_checkpoint(self.model, checkpoint_path, epoch=1)

            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )

            checkpoint["target_mean"] = torch.zeros(
                (1, 1, 3, 1, 1),
            )
            checkpoint["target_std"] = torch.ones(
                (1, 1, 3, 1, 1),
            )

            torch.save(checkpoint, checkpoint_path)

            result = evaluate_checkpoint(
                checkpoint_path,
                self.dataset.test,
            )

            self.assertEqual(
                result.samples_evaluated,
                len(self.dataset.test),
            )
            self.assertEqual(
                result.model_version,
                checkpoint["model_version"],
            )
            self.assertIn("water_level_m", result.fno_metrics)
            self.assertIn("water_level_m", result.baseline_metrics)
