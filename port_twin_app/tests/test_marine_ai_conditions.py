from types import SimpleNamespace
from uuid import uuid4

from django.test import SimpleTestCase

from port_twin_app.models import MarinePrediction
from port_twin_app.services.marine_ai.conditions import (
    build_marine_condition_advisory,
)


class MarineConditionAdvisoryTests(SimpleTestCase):
    def setUp(self):
        self.prediction = SimpleNamespace(
            id=uuid4(),
            status=MarinePrediction.Status.COMPLETED,
            grid={"nx": 2, "ny": 2},
            forecast={
                "start_time": "2026-11-25T08:00:00+00:00",
                "steps": 6,
                "step_seconds": 3600,
            },
            model_version="fno-test-1",
            contract_version="2.0",
            result_data={
                "water_level_m": [
                    [[1.0 + step, 2.0 + step], [3.0 + step, 4.0 + step]]
                    for step in range(6)
                ],
                "current_u_ms": [[[3.0, 0.0], [0.0, 0.0]] for _ in range(6)],
                "current_v_ms": [[[4.0, 2.0], [0.0, 0.0]] for _ in range(6)],
            },
        )

    def test_builds_six_hourly_grid_summaries(self):
        result = build_marine_condition_advisory(self.prediction)

        self.assertEqual(len(result["hourly_conditions"]), 6)
        self.assertEqual(result["scope"], "grid_level")

        first_hour = result["hourly_conditions"][0]

        self.assertEqual(first_hour["step"], 1)
        self.assertEqual(
            first_hour["start_time"],
            "2026-11-25T08:00:00+00:00",
        )
        self.assertEqual(
            first_hour["end_time"],
            "2026-11-25T09:00:00+00:00",
        )

        self.assertEqual(first_hour["mean_water_level_m"], 2.5)
        self.assertEqual(first_hour["minimum_water_level_m"], 1.0)
        self.assertEqual(first_hour["maximum_water_level_m"], 4.0)
        self.assertAlmostEqual(first_hour["mean_current_speed_ms"], 1.75)
        self.assertEqual(first_hour["maximum_current_speed_ms"], 5.0)

    def test_calculates_different_summaries_for_each_hour(self):
        result = build_marine_condition_advisory(self.prediction)

        first = result["hourly_conditions"][0]
        second = result["hourly_conditions"][1]

        self.assertEqual(first["mean_water_level_m"], 2.5)
        self.assertEqual(second["mean_water_level_m"], 3.5)

    def test_rejects_incomplete_prediction(self):
        self.prediction.status = MarinePrediction.Status.PENDING

        with self.assertRaisesMessage(ValueError, "must be completed"):
            build_marine_condition_advisory(self.prediction)

    def test_rejects_mismatched_grid_dimensions(self):
        self.prediction.grid["nx"] = 3

        with self.assertRaisesMessage(ValueError, "do not match its grid"):
            build_marine_condition_advisory(self.prediction)

    def test_rejects_invalid_forecast_values(self):
        self.prediction.result_data["water_level_m"][0][0][0] = float("nan")

        with self.assertRaisesMessage(ValueError, "non-finite or invalid"):
            build_marine_condition_advisory(self.prediction)

    def test_rejects_different_output_shapes(self):
        self.prediction.result_data["current_v_ms"][0] = [[1.0]]

        with self.assertRaisesMessage(
            ValueError,
            "current_v_ms contains inconsistent forecast dimensions",
        ):
            build_marine_condition_advisory(self.prediction)

    def test_rejects_wrong_forecast_step_count(self):
        self.prediction.forecast["steps"] = 5

        with self.assertRaisesMessage(ValueError, "six hourly forecast steps"):
            build_marine_condition_advisory(self.prediction)

    def test_rejects_timezone_naive_start_time(self):
        self.prediction.forecast["start_time"] = "2026-11-25T08:00:00"

        with self.assertRaisesMessage(ValueError, "include a timezone"):
            build_marine_condition_advisory(self.prediction)
