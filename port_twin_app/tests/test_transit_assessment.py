from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import xarray as xr
from django.test import SimpleTestCase

from port_twin_app.services.marine_data.shallow_water import (
    ShallowWaterConfig,
)
from port_twin_app.services.marine_data.transit.assessment import (
    _calculate_sample_metrics,
    _group_passing_samples,
    run_transit_assessment,
)
from port_twin_app.services.marine_data.transit.config import (
    DraftClass,
    TransitAssessmentConfig,
)


def synthetic_dataset():
    """Create a small dataset with predictable UKC and current values."""

    water_mask = np.array(
        [
            [True, True],
            [True, False],
        ],
    )

    bathymetry = np.array(
        [
            [20.0, 15.0],
            [10.0, np.nan],
        ],
    )

    water_levels = np.array(
        [
            [[0.0, 0.0], [0.0, np.nan]],
            [[1.0, 0.5], [-0.5, np.nan]],
            [[0.0, 0.0], [0.0, np.nan]],
        ],
    )

    current_u = np.array(
        [
            [[0.1, 0.2], [0.3, np.nan]],
            [[0.2, 0.4], [0.6, np.nan]],
            [[0.1, 0.1], [0.1, np.nan]],
        ],
    )

    current_v = np.zeros_like(current_u)

    return xr.Dataset(
        data_vars={
            "water_level_m": (
                ("time_seconds", "northing", "easting"),
                water_levels,
            ),
            "current_u_ms": (
                ("time_seconds", "northing", "easting"),
                current_u,
            ),
            "current_v_ms": (
                ("time_seconds", "northing", "easting"),
                current_v,
            ),
            "bathymetry_m": (
                ("northing", "easting"),
                bathymetry,
            ),
            "water_mask": (
                ("northing", "easting"),
                water_mask,
            ),
        },
        coords={
            "time_seconds": [0.0, 3600.0, 7200.0],
            "northing": [0.0, 250.0],
            "easting": [0.0, 250.0],
        },
        attrs={"data_classification": "simulated"},
    )


class TransitAssessmentConfigTests(SimpleTestCase):
    def test_default_draft_classes_and_thresholds(self):
        config = TransitAssessmentConfig()

        self.assertEqual(
            [
                (
                    draft.name,
                    draft.representative_draft_m,
                    draft.minimum_ukc_m,
                )
                for draft in config.draft_classes
            ],
            [
                ("small", 5.0, 1.0),
                ("medium", 10.0, 1.5),
                ("deep_draft", 15.0, 2.0),
            ],
        )
        self.assertEqual(config.maximum_current_speed_ms, 0.75)
        self.assertEqual(config.maximum_wind_speed_ms, 10.0)

    def test_configuration_is_json_serializable(self):
        config = TransitAssessmentConfig()

        self.assertEqual(
            config.to_dict()["maximum_current_speed_ms"],
            0.75,
        )
        self.assertEqual(
            config.to_dict()["draft_classes"][0]["name"],
            "small",
        )

    def test_rejects_empty_draft_classes(self):
        with self.assertRaisesMessage(
            ValueError,
            "At least one draft class is required.",
        ):
            TransitAssessmentConfig(draft_classes=())

    def test_rejects_duplicate_draft_class_names(self):
        with self.assertRaisesMessage(
            ValueError,
            "Draft class names must be unique.",
        ):
            TransitAssessmentConfig(
                draft_classes=(
                    DraftClass("small", 5.0, 1.0),
                    DraftClass("small", 6.0, 1.2),
                )
            )

    def test_rejects_invalid_draft_values(self):
        with self.assertRaises(ValueError):
            DraftClass("small", 0.0, 1.0)

        with self.assertRaises(ValueError):
            DraftClass("small", 5.0, -1.0)

    def test_rejects_negative_environmental_thresholds(self):
        with self.assertRaises(ValueError):
            TransitAssessmentConfig(maximum_current_speed_ms=-0.1)

        with self.assertRaises(ValueError):
            TransitAssessmentConfig(maximum_wind_speed_ms=-0.1)


class TransitMetricTests(SimpleTestCase):
    def test_calculates_minimum_ukc_and_maximum_current(self):
        samples = _calculate_sample_metrics(
            synthetic_dataset(),
            draft_m=5.0,
        )

        self.assertEqual(len(samples), 3)

        self.assertAlmostEqual(samples[0]["minimum_ukc_m"], 5.0)
        self.assertAlmostEqual(
            samples[0]["maximum_current_speed_ms"],
            0.3,
        )

        self.assertAlmostEqual(samples[1]["minimum_ukc_m"], 4.5)
        self.assertAlmostEqual(
            samples[1]["maximum_current_speed_ms"],
            0.6,
        )

        self.assertEqual(samples[2]["elapsed_seconds"], 7200.0)

    def test_ignores_land_cells(self):
        dataset = synthetic_dataset()

        samples = _calculate_sample_metrics(dataset, draft_m=5.0)

        self.assertTrue(all(np.isfinite(sample["minimum_ukc_m"]) for sample in samples))

    def test_rejects_non_finite_water_level(self):
        dataset = synthetic_dataset()
        values = dataset["water_level_m"].values.copy()
        values[0, 0, 0] = np.nan
        dataset["water_level_m"].values[:] = values

        with self.assertRaisesMessage(
            ValueError,
            "water level output contains non-finite water values.",
        ):
            _calculate_sample_metrics(dataset, draft_m=5.0)

    def test_rejects_non_increasing_output_times(self):
        dataset = synthetic_dataset()
        dataset = dataset.assign_coords(
            time_seconds=[0.0, 3600.0, 3600.0],
        )

        with self.assertRaisesMessage(
            ValueError,
            "strictly increasing",
        ):
            _calculate_sample_metrics(dataset, draft_m=5.0)

    def test_rejects_empty_water_mask(self):
        dataset = synthetic_dataset()
        dataset["water_mask"].values[:] = False

        with self.assertRaisesMessage(
            ValueError,
            "no computational water cells",
        ):
            _calculate_sample_metrics(dataset, draft_m=5.0)


class TransitWindowTests(SimpleTestCase):
    def test_groups_consecutive_passing_samples(self):
        samples = [
            {"elapsed_seconds": 0.0, "passes": True},
            {"elapsed_seconds": 3600.0, "passes": True},
            {"elapsed_seconds": 7200.0, "passes": False},
            {"elapsed_seconds": 10800.0, "passes": True},
            {"elapsed_seconds": 14400.0, "passes": True},
        ]

        windows = _group_passing_samples(samples)

        self.assertEqual(len(windows), 2)

        self.assertEqual(
            windows[0],
            {
                "start_elapsed_seconds": 0.0,
                "end_elapsed_seconds": 3600.0,
                "span_seconds": 3600.0,
                "sample_count": 2,
                "sample_times_seconds": [0.0, 3600.0],
            },
        )

        self.assertEqual(windows[1]["start_elapsed_seconds"], 10800.0)
        self.assertEqual(windows[1]["end_elapsed_seconds"], 14400.0)

    def test_single_passing_sample_has_zero_span(self):
        windows = _group_passing_samples(
            [{"elapsed_seconds": 3600.0, "passes": True}],
        )

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["span_seconds"], 0.0)
        self.assertEqual(windows[0]["sample_count"], 1)

    def test_returns_no_windows_when_all_samples_fail(self):
        windows = _group_passing_samples(
            [
                {"elapsed_seconds": 0.0, "passes": False},
                {"elapsed_seconds": 3600.0, "passes": False},
            ],
        )

        self.assertEqual(windows, [])

    def test_returns_no_windows_for_empty_samples(self):
        self.assertEqual(_group_passing_samples([]), [])


class TransitAssessmentIntegrationTests(SimpleTestCase):
    @patch(
        "port_twin_app.services.marine_data.transit.assessment."
        "run_shallow_water_simulation"
    )
    @patch("port_twin_app.services.marine_data.transit.assessment._load_domain")
    @patch(
        "port_twin_app.services.marine_data.transit.assessment."
        "get_shallow_water_scenarios"
    )
    def test_returns_advisory_assessment_with_provenance(
        self,
        mock_get_scenarios,
        mock_load_domain,
        mock_solver,
    ):
        config = ShallowWaterConfig(
            wind_u_ms=3.0,
            wind_v_ms=4.0,
            duration_seconds=7200,
            output_interval_seconds=3600,
        )

        scenario = SimpleNamespace(
            name="synthetic_scenario",
            description="Synthetic test scenario.",
            config=config,
            dataset_version="test",
        )

        domain = SimpleNamespace(
            projected_crs="EPSG:32737",
            grid_spacing_m=250.0,
            total_water_cells=3,
            excluded_water_cells=0,
            source_file="synthetic.nc",
            source_sha256="a" * 64,
            source_dataset_version="GEBCO_test",
            source_vertical_reference="mean_sea_level",
        )

        mock_get_scenarios.return_value = (scenario,)
        mock_load_domain.return_value = domain
        mock_solver.return_value = synthetic_dataset()

        assessment_config = TransitAssessmentConfig(
            draft_classes=(DraftClass("small", 5.0, 1.0),),
            maximum_current_speed_ms=0.75,
            maximum_wind_speed_ms=10.0,
        )

        result = run_transit_assessment(
            scenario_name="synthetic_scenario",
            domain_name="approaches",
            assessment_config=assessment_config,
        )

        self.assertEqual(result["assessment_type"], "experimental")
        self.assertTrue(result["advisory_only"])
        self.assertEqual(result["data_classification"], "simulated")

        self.assertEqual(result["scenario"]["name"], "synthetic_scenario")
        self.assertEqual(result["domain"]["projected_crs"], "EPSG:32737")

        self.assertAlmostEqual(result["forcing"]["wind_speed_ms"], 5.0)
        self.assertEqual(
            result["forcing"]["wind_source"],
            "constant_shallow_water_scenario_forcing",
        )

        self.assertEqual(
            result["bathymetry_provenance"]["source_sha256"],
            "a" * 64,
        )
        self.assertFalse(
            result["bathymetry_provenance"]["vertical_datum_correction_applied"]
        )

        self.assertEqual(len(result["draft_classes"]), 1)
        self.assertEqual(
            len(result["draft_classes"][0]["samples"]),
            3,
        )

        self.assertIn("limitations", result)
        self.assertTrue(result["limitations"])

        mock_get_scenarios.assert_called_once_with(
            ("synthetic_scenario",),
        )
        mock_load_domain.assert_called_once_with("approaches")
        mock_solver.assert_called_once_with(domain, config)

    @patch(
        "port_twin_app.services.marine_data.transit.assessment."
        "run_shallow_water_simulation"
    )
    @patch("port_twin_app.services.marine_data.transit.assessment._load_domain")
    @patch(
        "port_twin_app.services.marine_data.transit.assessment."
        "get_shallow_water_scenarios"
    )
    def test_wind_above_limit_fails_every_sample(
        self,
        mock_get_scenarios,
        mock_load_domain,
        mock_solver,
    ):
        scenario = SimpleNamespace(
            name="windy",
            description="Wind threshold test.",
            config=ShallowWaterConfig(
                wind_u_ms=10.0,
                wind_v_ms=5.0,
            ),
            dataset_version="test",
        )

        mock_get_scenarios.return_value = (scenario,)
        mock_load_domain.return_value = SimpleNamespace(
            projected_crs="EPSG:32737",
            grid_spacing_m=250.0,
            source_vertical_reference="height_above_mean_sea_level",
            source_dataset_version="GEBCO_2026",
            source_file="synthetic_bathymetry.nc",
            source_sha256="test-checksum",
            total_water_cells=3,
            excluded_water_cells=0,
        )
        mock_solver.return_value = synthetic_dataset()

        config = TransitAssessmentConfig(
            draft_classes=(DraftClass("small", 5.0, 1.0),),
            maximum_wind_speed_ms=10.0,
        )

        result = run_transit_assessment(
            "windy",
            assessment_config=config,
        )

        samples = result["draft_classes"][0]["samples"]

        self.assertTrue(
            all(
                "wind_above_threshold" in sample["failure_reasons"]
                for sample in samples
            )
        )
        self.assertEqual(result["draft_classes"][0]["windows"], [])

    @patch(
        "port_twin_app.services.marine_data.transit.assessment."
        "run_shallow_water_simulation"
    )
    @patch("port_twin_app.services.marine_data.transit.assessment._load_domain")
    @patch(
        "port_twin_app.services.marine_data.transit.assessment."
        "get_shallow_water_scenarios"
    )
    def test_current_above_limit_fails_every_sample(
        self,
        mock_get_scenarios,
        mock_load_domain,
        mock_solver,
    ):
        scenario = SimpleNamespace(
            name="strong_current",
            description="Current threshold test.",
            config=ShallowWaterConfig(
                wind_u_ms=0.0,
                wind_v_ms=0.0,
            ),
            dataset_version="test",
        )

        domain = SimpleNamespace(
            projected_crs="EPSG:32737",
            grid_spacing_m=250.0,
            source_vertical_reference="height_above_mean_sea_level",
            source_dataset_version="GEBCO_2026",
            source_file="synthetic_bathymetry.nc",
            source_sha256="test-checksum",
            total_water_cells=3,
            excluded_water_cells=0,
        )

        dataset = synthetic_dataset()
        dataset["current_u_ms"].values[:, :2, :] = 0.9

        mock_get_scenarios.return_value = (scenario,)
        mock_load_domain.return_value = domain
        mock_solver.return_value = dataset

        config = TransitAssessmentConfig(
            draft_classes=(DraftClass("small", 5.0, 1.0),),
            maximum_current_speed_ms=0.75,
            maximum_wind_speed_ms=10.0,
        )

        result = run_transit_assessment(
            "strong_current",
            assessment_config=config,
        )

        samples = result["draft_classes"][0]["samples"]

        self.assertTrue(
            all(
                "current_above_threshold" in sample["failure_reasons"]
                for sample in samples
            )
        )
        self.assertEqual(result["draft_classes"][0]["windows"], [])
