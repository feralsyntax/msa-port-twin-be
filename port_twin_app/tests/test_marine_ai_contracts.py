from django.test import SimpleTestCase

from port_twin_app.services.marine_ai.contracts import (
    CONTRACT_VERSION,
    MarineForecastConfig,
    MarineGrid,
    MarinePredictionRequest,
    MarinePredictionResult,
)


class MarineGridTests(SimpleTestCase):
    def test_valid_grid_is_accepted(self):
        grid = MarineGrid(
            west=39.5,
            south=-4.2,
            east=39.8,
            north=-3.9,
            nx=3,
            ny=2,
        )

        self.assertEqual(grid.crs, "EPSG:4326")
        self.assertEqual(grid.nx, 3)
        self.assertEqual(grid.ny, 2)

    def test_invalid_longitude_bounds_are_rejected(self):
        with self.assertRaises(ValueError):
            MarineGrid(
                west=40,
                south=-4.2,
                east=39.8,
                north=-3.9,
                nx=3,
                ny=2,
            )

    def test_invalid_latitude_bounds_are_rejected(self):
        with self.assertRaises(ValueError):
            MarineGrid(
                west=39.5,
                south=-91,
                east=39.8,
                north=-3.9,
                nx=3,
                ny=2,
            )

    def test_grid_dimensions_must_be_at_least_two(self):
        with self.assertRaises(ValueError):
            MarineGrid(
                west=39.5,
                south=-4.2,
                east=39.8,
                north=-3.9,
                nx=1,
                ny=2,
            )

    def test_unsupported_coordinate_reference_system_is_rejected(self):
        with self.assertRaises(ValueError):
            MarineGrid(
                west=39.5,
                south=-4.2,
                east=39.8,
                north=-3.9,
                nx=3,
                ny=2,
                crs="EPSG:3857",
            )


class MarineForecastConfigTests(SimpleTestCase):
    def test_valid_forecast_is_accepted(self):
        forecast = MarineForecastConfig(
            start_time="2026-10-20T10:00:00Z",
            steps=4,
            step_seconds=900,
        )

        self.assertEqual(forecast.steps, 4)
        self.assertEqual(forecast.step_seconds, 900)

    def test_timezone_is_required(self):
        with self.assertRaises(ValueError):
            MarineForecastConfig(
                start_time="2026-10-20T10:00:00",
                steps=4,
                step_seconds=900,
            )

    def test_invalid_timestamp_is_rejected(self):
        with self.assertRaises(ValueError):
            MarineForecastConfig(
                start_time="not-a-timestamp",
                steps=4,
                step_seconds=900,
            )

    def test_steps_must_be_positive(self):
        with self.assertRaises(ValueError):
            MarineForecastConfig(
                start_time="2026-10-20T10:00:00Z",
                steps=0,
                step_seconds=900,
            )

    def test_step_interval_must_be_positive(self):
        with self.assertRaises(ValueError):
            MarineForecastConfig(
                start_time="2026-10-20T10:00:00Z",
                steps=4,
                step_seconds=0,
            )


class MarinePredictionRequestTests(SimpleTestCase):
    def setUp(self):
        self.grid = MarineGrid(
            west=39.5,
            south=-4.2,
            east=39.8,
            north=-3.9,
            nx=2,
            ny=2,
        )
        self.forecast = MarineForecastConfig(
            start_time="2026-10-20T10:00:00Z",
            steps=2,
            step_seconds=900,
        )
        self.valid_data = {
            "grid": self.grid,
            "forecast": self.forecast,
            "bathymetry_m": [
                [10.0, 11.0],
                [12.0, 13.0],
            ],
            "initial_water_level_m": [
                [1.0, 1.1],
                [1.2, 1.3],
            ],
            "wind_u_ms": [
                [[2.0, 2.1], [2.2, 2.3]],
                [[2.4, 2.5], [2.6, 2.7]],
            ],
            "wind_v_ms": [
                [[1.0, 1.1], [1.2, 1.3]],
                [[1.4, 1.5], [1.6, 1.7]],
            ],
            "boundary_conditions": {},
        }

    def test_valid_prediction_request_is_accepted(self):
        request = MarinePredictionRequest(**self.valid_data)

        self.assertEqual(request.grid.nx, 2)
        self.assertEqual(len(request.wind_u_ms), 2)

    def test_incorrect_bathymetry_shape_is_rejected(self):
        data = {
            **self.valid_data,
            "bathymetry_m": [[10.0, 11.0]],
        }

        with self.assertRaises(ValueError):
            MarinePredictionRequest(**data)

    def test_incorrect_initial_water_level_shape_is_rejected(self):
        data = {
            **self.valid_data,
            "initial_water_level_m": [[1.0, 1.1], [1.2]],
        }

        with self.assertRaises(ValueError):
            MarinePredictionRequest(**data)

    def test_incorrect_wind_time_dimension_is_rejected(self):
        data = {
            **self.valid_data,
            "wind_u_ms": [
                [[2.0, 2.1], [2.2, 2.3]],
            ],
        }

        with self.assertRaises(ValueError):
            MarinePredictionRequest(**data)

    def test_incorrect_wind_grid_dimensions_are_rejected(self):
        data = {
            **self.valid_data,
            "wind_v_ms": [
                [[1.0, 1.1]],
                [[1.4, 1.5], [1.6, 1.7]],
            ],
        }

        with self.assertRaises(ValueError):
            MarinePredictionRequest(**data)

    def test_boundary_conditions_must_be_a_dictionary(self):
        data = {
            **self.valid_data,
            "boundary_conditions": [],
        }

        with self.assertRaises(ValueError):
            MarinePredictionRequest(**data)


class MarinePredictionResultTests(SimpleTestCase):
    def setUp(self):
        self.valid_data = {
            "water_level_m": [
                [[1.0, 1.1], [1.2, 1.3]],
                [[1.4, 1.5], [1.6, 1.7]],
            ],
            "current_u_ms": [
                [[0.1, 0.2], [0.3, 0.4]],
                [[0.5, 0.6], [0.7, 0.8]],
            ],
            "current_v_ms": [
                [[0.2, 0.3], [0.4, 0.5]],
                [[0.6, 0.7], [0.8, 0.9]],
            ],
            "model_version": "initial-0.1",
        }

    def test_valid_prediction_result_is_accepted(self):
        result = MarinePredictionResult(**self.valid_data)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.contract_version, CONTRACT_VERSION)
        self.assertEqual(result.model_version, "initial-0.1")

    def test_prediction_fields_must_have_matching_shapes(self):
        data = {
            **self.valid_data,
            "current_u_ms": [[[0.1, 0.2]]],
        }

        with self.assertRaises(ValueError):
            MarinePredictionResult(**data)

    def test_empty_model_version_is_rejected(self):
        data = {
            **self.valid_data,
            "model_version": " ",
        }

        with self.assertRaises(ValueError):
            MarinePredictionResult(**data)

    def test_empty_prediction_field_is_rejected(self):
        data = {
            **self.valid_data,
            "water_level_m": [],
        }

        with self.assertRaises(ValueError):
            MarinePredictionResult(**data)
