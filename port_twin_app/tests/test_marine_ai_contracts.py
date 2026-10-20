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

    def test_non_finite_coordinates_are_rejected(self):
        with self.assertRaises(ValueError):
            MarineGrid(
                west=float("nan"),
                south=-4.2,
                east=39.8,
                north=-3.9,
                nx=2,
                ny=2,
            )

    def test_infinite_coordinates_are_rejected(self):
        with self.assertRaises(ValueError):
            MarineGrid(
                west=39.5,
                south=-4.2,
                east=float("inf"),
                north=-3.9,
                nx=2,
                ny=2,
            )

    def test_fractional_grid_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            MarineGrid(
                west=39.5,
                south=-4.2,
                east=39.8,
                north=-3.9,
                nx=2.5,
                ny=2,
            )

    def test_boolean_grid_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            MarineGrid(
                west=39.5,
                south=-4.2,
                east=39.8,
                north=-3.9,
                nx=True,
                ny=2,
            )

    def test_boolean_coordinates_are_rejected(self):
        with self.assertRaises(ValueError):
            MarineGrid(
                west=True,
                south=-4.2,
                east=39.8,
                north=-3.9,
                nx=2,
                ny=2,
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

    def test_non_string_timestamp_is_rejected(self):
        with self.assertRaises(ValueError):
            MarineForecastConfig(
                start_time=None,
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

    def test_fractional_steps_are_rejected(self):
        with self.assertRaises(ValueError):
            MarineForecastConfig(
                start_time="2026-10-20T10:00:00Z",
                steps=2.5,
                step_seconds=900,
            )

    def test_boolean_steps_are_rejected(self):
        with self.assertRaises(ValueError):
            MarineForecastConfig(
                start_time="2026-10-20T10:00:00Z",
                steps=True,
                step_seconds=900,
            )

    def test_fractional_step_interval_is_rejected(self):
        with self.assertRaises(ValueError):
            MarineForecastConfig(
                start_time="2026-10-20T10:00:00Z",
                steps=4,
                step_seconds=900.5,
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
            "boundary_conditions": {
                "west": {
                    "water_level_m": [1.0, 1.1],
                },
            },
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
            "initial_water_level_m": [
                [1.0, 1.1],
                [1.2],
            ],
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

    def test_non_list_grid_field_is_rejected(self):
        data = {
            **self.valid_data,
            "bathymetry_m": None,
        }

        with self.assertRaises(ValueError):
            MarinePredictionRequest(**data)

    def test_non_list_time_field_is_rejected(self):
        data = {
            **self.valid_data,
            "wind_u_ms": None,
        }

        with self.assertRaises(ValueError):
            MarinePredictionRequest(**data)

    def test_nan_wind_values_are_rejected(self):
        data = {
            **self.valid_data,
            "wind_u_ms": [
                [[float("nan"), 2.1], [2.2, 2.3]],
                [[2.4, 2.5], [2.6, 2.7]],
            ],
        }

        with self.assertRaises(ValueError):
            MarinePredictionRequest(**data)

    def test_infinite_wind_values_are_rejected(self):
        data = {
            **self.valid_data,
            "wind_v_ms": [
                [[1.0, 1.1], [1.2, 1.3]],
                [[1.4, float("inf")], [1.6, 1.7]],
            ],
        }

        with self.assertRaises(ValueError):
            MarinePredictionRequest(**data)

    def test_boolean_bathymetry_values_are_rejected(self):
        data = {
            **self.valid_data,
            "bathymetry_m": [
                [True, 11.0],
                [12.0, 13.0],
            ],
        }

        with self.assertRaises(ValueError):
            MarinePredictionRequest(**data)

    def test_boolean_initial_water_level_is_rejected(self):
        data = {
            **self.valid_data,
            "initial_water_level_m": [
                [1.0, False],
                [1.2, 1.3],
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

    def test_empty_boundary_conditions_are_rejected(self):
        data = {
            **self.valid_data,
            "boundary_conditions": {},
        }

        with self.assertRaises(ValueError):
            MarinePredictionRequest(**data)

    def test_unknown_boundary_edge_is_rejected(self):
        data = {
            **self.valid_data,
            "boundary_conditions": {
                "upper": {
                    "water_level_m": [1.0, 1.1],
                },
            },
        }

        with self.assertRaises(ValueError):
            MarinePredictionRequest(**data)

    def test_boundary_condition_must_be_a_dictionary(self):
        data = {
            **self.valid_data,
            "boundary_conditions": {
                "west": [1.0, 1.1],
            },
        }

        with self.assertRaises(ValueError):
            MarinePredictionRequest(**data)

    def test_boundary_must_define_exactly_one_variable(self):
        data = {
            **self.valid_data,
            "boundary_conditions": {
                "west": {
                    "water_level_m": [1.0, 1.1],
                    "normal_velocity_ms": [0.1, 0.2],
                },
            },
        }

        with self.assertRaises(ValueError):
            MarinePredictionRequest(**data)

    def test_unsupported_boundary_variable_is_rejected(self):
        data = {
            **self.valid_data,
            "boundary_conditions": {
                "west": {
                    "temperature": [27.0, 27.1],
                },
            },
        }

        with self.assertRaises(ValueError):
            MarinePredictionRequest(**data)

    def test_boundary_series_must_match_forecast_steps(self):
        data = {
            **self.valid_data,
            "boundary_conditions": {
                "west": {
                    "water_level_m": [1.0],
                },
            },
        }

        with self.assertRaises(ValueError):
            MarinePredictionRequest(**data)

    def test_boundary_values_must_be_finite(self):
        data = {
            **self.valid_data,
            "boundary_conditions": {
                "west": {
                    "water_level_m": [1.0, float("inf")],
                },
            },
        }

        with self.assertRaises(ValueError):
            MarinePredictionRequest(**data)

    def test_boundary_boolean_values_are_rejected(self):
        data = {
            **self.valid_data,
            "boundary_conditions": {
                "west": {
                    "water_level_m": [1.0, True],
                },
            },
        }

        with self.assertRaises(ValueError):
            MarinePredictionRequest(**data)

    def test_invalid_grid_object_is_rejected(self):
        data = {
            **self.valid_data,
            "grid": None,
        }

        with self.assertRaises(ValueError):
            MarinePredictionRequest(**data)

    def test_invalid_forecast_object_is_rejected(self):
        data = {
            **self.valid_data,
            "forecast": None,
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

    def test_non_string_model_version_is_rejected(self):
        data = {
            **self.valid_data,
            "model_version": None,
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

    def test_malformed_prediction_array_is_rejected(self):
        data = {
            **self.valid_data,
            "water_level_m": None,
        }

        with self.assertRaises(ValueError):
            MarinePredictionResult(**data)

    def test_nan_water_level_is_rejected(self):
        data = {
            **self.valid_data,
            "water_level_m": [
                [[float("nan"), 1.1], [1.2, 1.3]],
                [[1.4, 1.5], [1.6, 1.7]],
            ],
        }

        with self.assertRaises(ValueError):
            MarinePredictionResult(**data)

    def test_infinite_current_values_are_rejected(self):
        data = {
            **self.valid_data,
            "current_u_ms": [
                [[float("inf"), 0.2], [0.3, 0.4]],
                [[0.5, 0.6], [0.7, 0.8]],
            ],
        }

        with self.assertRaises(ValueError):
            MarinePredictionResult(**data)

    def test_boolean_current_values_are_rejected(self):
        data = {
            **self.valid_data,
            "current_v_ms": [
                [[True, 0.3], [0.4, 0.5]],
                [[0.6, 0.7], [0.8, 0.9]],
            ],
        }

        with self.assertRaises(ValueError):
            MarinePredictionResult(**data)

    def test_unsupported_contract_version_is_rejected(self):
        data = {
            **self.valid_data,
            "contract_version": "99.0",
        }

        with self.assertRaises(ValueError):
            MarinePredictionResult(**data)

    def test_non_success_status_is_rejected(self):
        data = {
            **self.valid_data,
            "status": "failed",
        }

        with self.assertRaises(ValueError):
            MarinePredictionResult(**data)

    def test_result_must_match_request_dimensions(self):
        result = MarinePredictionResult(**self.valid_data)

        request = MarinePredictionRequest(
            grid=MarineGrid(
                west=39.5,
                south=-4.2,
                east=39.8,
                north=-3.9,
                nx=3,
                ny=2,
            ),
            forecast=MarineForecastConfig(
                start_time="2026-10-20T10:00:00Z",
                steps=2,
                step_seconds=900,
            ),
            bathymetry_m=[
                [10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0],
            ],
            initial_water_level_m=[
                [1.0, 1.1, 1.2],
                [1.3, 1.4, 1.5],
            ],
            wind_u_ms=[
                [[2.0, 2.1, 2.2], [2.3, 2.4, 2.5]],
                [[2.6, 2.7, 2.8], [2.9, 3.0, 3.1]],
            ],
            wind_v_ms=[
                [[1.0, 1.1, 1.2], [1.3, 1.4, 1.5]],
                [[1.6, 1.7, 1.8], [1.9, 2.0, 2.1]],
            ],
            boundary_conditions={
                "west": {
                    "water_level_m": [1.0, 1.1],
                },
            },
        )

        with self.assertRaises(ValueError):
            result.validate_against(request)

    def test_result_matching_request_dimensions_is_accepted(self):
        result = MarinePredictionResult(**self.valid_data)

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
                start_time="2026-10-20T10:00:00Z",
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
                "west": {
                    "water_level_m": [1.0, 1.1],
                },
            },
        )

        self.assertIsNone(result.validate_against(request))
