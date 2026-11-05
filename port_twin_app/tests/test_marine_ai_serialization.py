"""Tests for prepared marine AI dataset serialization."""

import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from django.test import SimpleTestCase

from port_twin_app.services.marine_ai.contracts import (
    MarineForecastConfig,
    MarineGrid,
    MarinePredictionRequest,
    MarinePredictionResult,
)
from port_twin_app.services.marine_ai.dataset import (
    MarineDatasetSample,
    MarineDatasetSplit,
)
from port_twin_app.services.marine_ai.preparation import PreparedMarineDataset
from port_twin_app.services.marine_ai.serialization import (
    FORMAT_VERSION,
    load_prepared_dataset,
    save_prepared_dataset,
)


class MarineDatasetSerializationTests(SimpleTestCase):
    """Tests for saving and loading prepared marine datasets."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.archive_path = Path(self.temp_dir.name) / "marine_dataset.npz"

        self.prepared = self._build_prepared_dataset()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _build_sample(self, index):
        grid = MarineGrid(
            west=40.0,
            south=-4.0,
            east=40.1,
            north=-3.9,
            nx=2,
            ny=2,
        )

        start_time = (
            datetime(2026, 10, 1, tzinfo=UTC) + timedelta(hours=index)
        ).isoformat()

        bathymetry = [
            [100.0, 150.0],
            [200.0, 250.0],
        ]

        initial_water_level = [
            [0.1 + index, 0.2 + index],
            [0.3 + index, 0.4 + index],
        ]

        initial_current_u = [
            [0.01 + index, 0.02 + index],
            [0.03 + index, 0.04 + index],
        ]

        initial_current_v = [
            [0.05 + index, 0.06 + index],
            [0.07 + index, 0.08 + index],
        ]

        request = MarinePredictionRequest(
            grid=grid,
            forecast=MarineForecastConfig(start_time=start_time),
            bathymetry_m=bathymetry,
            initial_water_level_m=initial_water_level,
            initial_current_u_ms=initial_current_u,
            initial_current_v_ms=initial_current_v,
        )

        target_values = np.full((6, 2, 2), index + 1.0)

        target = MarinePredictionResult(
            water_level_m=target_values.tolist(),
            current_u_ms=(target_values * 0.1).tolist(),
            current_v_ms=(target_values * 0.2).tolist(),
            model_version="copernicus-observation",
        )

        return MarineDatasetSample(
            sample_id=f"sample-{index:03d}",
            request=request,
            target=target,
            source="Copernicus Marine + GEBCO",
            source_version="Copernicus:202406;GEBCO:GEBCO_2026",
            license_name="Copernicus Marine Service Licence; GEBCO public domain",
        )

    def _build_prepared_dataset(self):
        return PreparedMarineDataset(
            split=MarineDatasetSplit(
                train=(self._build_sample(0),),
                validation=(self._build_sample(1),),
                test=(self._build_sample(2),),
            ),
            metadata={
                "marine_source": "Copernicus Marine",
                "bathymetry_source": "GEBCO",
                "grid": {"nx": 2, "ny": 2},
                "vertical_reference_warning": "No datum transformation applied.",
            },
            tid_summary={
                40: 3,
                44: 1,
            },
        )

    def _save_and_load(self):
        save_prepared_dataset(self.prepared, self.archive_path)
        return load_prepared_dataset(self.archive_path)

    def _read_archive(self):
        with np.load(self.archive_path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}

    def _rewrite_archive(self, arrays):
        np.savez_compressed(self.archive_path, **arrays)

    def _read_manifest(self):
        metadata_path = self.archive_path.with_suffix(".json")
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    def _write_manifest(self, manifest):
        metadata_path = self.archive_path.with_suffix(".json")
        metadata_path.write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

    def test_save_creates_archive_and_manifest(self):
        archive_path, metadata_path = save_prepared_dataset(
            self.prepared,
            self.archive_path,
        )

        self.assertTrue(archive_path.is_file())
        self.assertTrue(metadata_path.is_file())

        manifest = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["format_version"], FORMAT_VERSION)
        self.assertEqual(manifest["partition_counts"]["train"], 1)
        self.assertEqual(manifest["partition_counts"]["validation"], 1)
        self.assertEqual(manifest["partition_counts"]["test"], 1)

    def test_round_trip_preserves_dataset(self):
        loaded = self._save_and_load()

        self.assertEqual(loaded.metadata, self.prepared.metadata)
        self.assertEqual(loaded.tid_summary, self.prepared.tid_summary)

        for partition in ("train", "validation", "test"):
            original_samples = getattr(self.prepared.split, partition)
            loaded_samples = getattr(loaded.split, partition)

            self.assertEqual(len(loaded_samples), len(original_samples))

            for original, restored in zip(
                original_samples,
                loaded_samples,
                strict=True,
            ):
                self.assertEqual(restored.sample_id, original.sample_id)
                self.assertEqual(restored.source, original.source)
                self.assertEqual(
                    restored.source_version,
                    original.source_version,
                )
                self.assertEqual(
                    restored.license_name,
                    original.license_name,
                )
                self.assertEqual(restored.request, original.request)
                self.assertEqual(restored.target, original.target)

    def test_round_trip_preserves_partition_assignments(self):
        loaded = self._save_and_load()

        self.assertEqual(
            [sample.sample_id for sample in loaded.split.train],
            ["sample-000"],
        )
        self.assertEqual(
            [sample.sample_id for sample in loaded.split.validation],
            ["sample-001"],
        )
        self.assertEqual(
            [sample.sample_id for sample in loaded.split.test],
            ["sample-002"],
        )

    def test_round_trip_preserves_tid_summary(self):
        loaded = self._save_and_load()

        self.assertEqual(
            loaded.tid_summary,
            {
                40: 3,
                44: 1,
            },
        )

    def test_round_trip_preserves_array_precision(self):
        self._save_and_load()

        arrays = self._read_archive()

        original_sample = self.prepared.split.train[0]

        np.testing.assert_array_equal(
            arrays["bathymetry_m"],
            np.asarray(original_sample.request.bathymetry_m),
        )

        np.testing.assert_array_equal(
            arrays["initial_states"][0, 0],
            np.asarray(original_sample.request.initial_water_level_m),
        )

        np.testing.assert_array_equal(
            arrays["targets"][0, :, 0],
            np.asarray(original_sample.target.water_level_m),
        )

    def test_save_rejects_invalid_prepared_dataset(self):
        with self.assertRaises(TypeError):
            save_prepared_dataset(object(), self.archive_path)

    def test_save_rejects_empty_partition(self):
        prepared = PreparedMarineDataset(
            split=MarineDatasetSplit(
                train=self.prepared.split.train,
                validation=(),
                test=self.prepared.split.test,
            ),
            metadata={},
            tid_summary={},
        )

        with self.assertRaises(ValueError):
            save_prepared_dataset(prepared, self.archive_path)

    def test_save_rejects_duplicate_sample_ids(self):
        sample = self._build_sample(0)

        prepared = PreparedMarineDataset(
            split=MarineDatasetSplit(
                train=(sample,),
                validation=(sample,),
                test=(self._build_sample(2),),
            ),
            metadata={},
            tid_summary={},
        )

        with self.assertRaisesRegex(ValueError, "unique"):
            save_prepared_dataset(prepared, self.archive_path)

    def test_load_rejects_missing_archive(self):
        with self.assertRaises(FileNotFoundError):
            load_prepared_dataset(self.archive_path)

    def test_load_rejects_missing_manifest(self):
        save_prepared_dataset(self.prepared, self.archive_path)
        self.archive_path.with_suffix(".json").unlink()

        with self.assertRaises(FileNotFoundError):
            load_prepared_dataset(self.archive_path)

    def test_load_rejects_unsupported_format_version(self):
        save_prepared_dataset(self.prepared, self.archive_path)

        manifest = self._read_manifest()
        manifest["format_version"] = FORMAT_VERSION + 1
        self._write_manifest(manifest)

        with self.assertRaisesRegex(ValueError, "format version"):
            load_prepared_dataset(self.archive_path)

    def test_load_rejects_incompatible_contract_version(self):
        save_prepared_dataset(self.prepared, self.archive_path)

        manifest = self._read_manifest()
        manifest["contract_version"] = "incompatible"
        self._write_manifest(manifest)

        with self.assertRaisesRegex(ValueError, "contract version"):
            load_prepared_dataset(self.archive_path)

    def test_load_rejects_missing_archive_array(self):
        save_prepared_dataset(self.prepared, self.archive_path)

        arrays = self._read_archive()
        del arrays["targets"]
        self._rewrite_archive(arrays)

        with self.assertRaisesRegex(ValueError, "missing arrays"):
            load_prepared_dataset(self.archive_path)

    def test_load_rejects_invalid_initial_state_shape(self):
        save_prepared_dataset(self.prepared, self.archive_path)

        arrays = self._read_archive()
        arrays["initial_states"] = arrays["initial_states"][:, :2]
        self._rewrite_archive(arrays)

        with self.assertRaisesRegex(ValueError, "initial states"):
            load_prepared_dataset(self.archive_path)

    def test_load_rejects_invalid_target_shape(self):
        save_prepared_dataset(self.prepared, self.archive_path)

        arrays = self._read_archive()
        arrays["targets"] = arrays["targets"][:, :5]
        self._rewrite_archive(arrays)

        with self.assertRaisesRegex(ValueError, "targets"):
            load_prepared_dataset(self.archive_path)

    def test_load_rejects_non_finite_values(self):
        save_prepared_dataset(self.prepared, self.archive_path)

        arrays = self._read_archive()
        arrays["bathymetry_m"][0, 0] = np.nan
        self._rewrite_archive(arrays)

        with self.assertRaisesRegex(ValueError, "non-finite"):
            load_prepared_dataset(self.archive_path)

    def test_load_rejects_duplicate_sample_ids(self):
        save_prepared_dataset(self.prepared, self.archive_path)

        arrays = self._read_archive()
        arrays["sample_ids"][1] = arrays["sample_ids"][0]
        self._rewrite_archive(arrays)

        with self.assertRaisesRegex(ValueError, "not unique"):
            load_prepared_dataset(self.archive_path)

    def test_load_rejects_unknown_partition(self):
        save_prepared_dataset(self.prepared, self.archive_path)

        arrays = self._read_archive()
        arrays["partitions"][0] = "unknown"
        self._rewrite_archive(arrays)

        with self.assertRaisesRegex(ValueError, "unknown partition"):
            load_prepared_dataset(self.archive_path)

    def test_load_rejects_partition_count_mismatch(self):
        save_prepared_dataset(self.prepared, self.archive_path)

        manifest = self._read_manifest()
        manifest["partition_counts"]["train"] = 99
        self._write_manifest(manifest)

        with self.assertRaisesRegex(ValueError, "partition count"):
            load_prepared_dataset(self.archive_path)
