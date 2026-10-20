from datetime import timedelta

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


class MarineDatasetSampleTests(SimpleTestCase):
    def make_sample(self, sample_id="sample-1", start_time=None):
        start_time = start_time or "2026-10-20T10:00:00Z"

        request = MarinePredictionRequest(
            grid=MarineGrid(
                west=39.5,
                south=-4.2,
                east=39.8,
                north=-3.9,
                nx=2,
                ny=2,
            ),
            forecast=MarineForecastConfig(
                start_time=start_time,
                steps=2,
                step_seconds=900,
            ),
            bathymetry_m=[
                [10.0, 11.0],
                [12.0, 13.0],
            ],
            initial_water_level_m=[
                [1.0, 1.1],
                [1.2, 1.3],
            ],
            wind_u_ms=[
                [[2.0, 2.1], [2.2, 2.3]],
                [[2.4, 2.5], [2.6, 2.7]],
            ],
            wind_v_ms=[
                [[1.0, 1.1], [1.2, 1.3]],
                [[1.4, 1.5], [1.6, 1.7]],
            ],
            boundary_conditions={
                "west": {"water_level_m": [1.0, 1.1]},
            },
        )

        target = MarinePredictionResult(
            water_level_m=[
                [[1.0, 1.1], [1.2, 1.3]],
                [[1.4, 1.5], [1.6, 1.7]],
            ],
            current_u_ms=[
                [[0.1, 0.2], [0.3, 0.4]],
                [[0.5, 0.6], [0.7, 0.8]],
            ],
            current_v_ms=[
                [[0.2, 0.3], [0.4, 0.5]],
                [[0.6, 0.7], [0.8, 0.9]],
            ],
            model_version="test-model-0.1",
        )

        return MarineDatasetSample(
            sample_id=sample_id,
            request=request,
            target=target,
            source="test-fixture",
            source_version="1.0",
            license_name="internal-test-data",
        )

    def test_valid_sample_is_accepted(self):
        sample = self.make_sample()

        self.assertEqual(sample.sample_id, "sample-1")
        self.assertEqual(sample.source, "test-fixture")

    def test_forecast_end_uses_forecast_duration(self):
        sample = self.make_sample()

        self.assertEqual(
            sample.forecast_end - sample.forecast_start,
            timedelta(minutes=30),
        )

    def test_empty_sample_id_is_rejected(self):
        with self.assertRaises(ValueError):
            self.make_sample(sample_id=" ")

    def test_missing_source_is_rejected(self):
        sample = self.make_sample()

        with self.assertRaises(ValueError):
            MarineDatasetSample(
                sample_id=sample.sample_id,
                request=sample.request,
                target=sample.target,
                source="",
                source_version=sample.source_version,
                license_name=sample.license_name,
            )

    def test_missing_license_metadata_is_rejected(self):
        sample = self.make_sample()

        with self.assertRaises(ValueError):
            MarineDatasetSample(
                sample_id=sample.sample_id,
                request=sample.request,
                target=sample.target,
                source=sample.source,
                source_version=sample.source_version,
                license_name=" ",
            )

    def test_mismatched_target_dimensions_are_rejected(self):
        sample = self.make_sample()

        mismatched_target = MarinePredictionResult(
            water_level_m=[[[1.0, 1.1], [1.2, 1.3]]],
            current_u_ms=[[[0.1, 0.2], [0.3, 0.4]]],
            current_v_ms=[[[0.2, 0.3], [0.4, 0.5]]],
            model_version="test-model-0.1",
        )

        with self.assertRaises(ValueError):
            MarineDatasetSample(
                sample_id="bad-sample",
                request=sample.request,
                target=mismatched_target,
                source="test-fixture",
                source_version="1.0",
                license_name="internal-test-data",
            )


class MarineDatasetSplitTests(MarineDatasetSampleTests):
    def make_samples(self, count=10, interval_minutes=60):
        start = self.make_sample().forecast_start

        return [
            self.make_sample(
                sample_id=f"sample-{index}",
                start_time=(
                    start + timedelta(minutes=index * interval_minutes)
                ).isoformat(),
            )
            for index in range(count)
        ]

    def test_dataset_is_split_chronologically(self):
        samples = self.make_samples()

        result = split_dataset(
            samples,
            train_ratio=0.6,
            validation_ratio=0.2,
        )

        self.assertTrue(result.train)
        self.assertTrue(result.validation)
        self.assertTrue(result.test)

        self.assertLessEqual(
            max(sample.forecast_start for sample in result.train),
            min(sample.forecast_start for sample in result.validation),
        )
        self.assertLessEqual(
            max(sample.forecast_start for sample in result.validation),
            min(sample.forecast_start for sample in result.test),
        )

    def test_samples_are_not_duplicated_between_splits(self):
        result = split_dataset(
            self.make_samples(),
            train_ratio=0.6,
            validation_ratio=0.2,
        )

        all_ids = [
            sample.sample_id
            for partition in (result.train, result.validation, result.test)
            for sample in partition
        ]

        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(result.total_samples, len(all_ids))

    def test_overlapping_forecast_windows_are_purged_at_boundaries(self):
        result = split_dataset(
            self.make_samples(interval_minutes=15),
            train_ratio=0.6,
            validation_ratio=0.2,
        )

        validation_start = min(sample.forecast_start for sample in result.validation)
        test_start = min(sample.forecast_start for sample in result.test)

        self.assertTrue(
            all(sample.forecast_end <= validation_start for sample in result.train)
        )
        self.assertTrue(
            all(sample.forecast_end <= test_start for sample in result.validation)
        )

    def test_duplicate_sample_ids_are_rejected(self):
        sample = self.make_sample()

        with self.assertRaises(ValueError):
            split_dataset(
                [sample, sample, self.make_sample(sample_id="sample-3")],
                train_ratio=0.34,
                validation_ratio=0.33,
            )

    def test_invalid_ratios_are_rejected(self):
        with self.assertRaises(ValueError):
            split_dataset(self.make_samples(), train_ratio=1.0)

    def test_ratios_cannot_sum_to_one(self):
        with self.assertRaises(ValueError):
            split_dataset(
                self.make_samples(),
                train_ratio=0.7,
                validation_ratio=0.3,
            )

    def test_insufficient_samples_are_rejected(self):
        with self.assertRaises(ValueError):
            split_dataset(self.make_samples(count=2))
