from django.test import SimpleTestCase

from port_twin_app.services.marine_ai.contracts import (
    CONTRACT_VERSION,
    FORECAST_STEP_SECONDS,
    FORECAST_STEPS,
    MarineForecastConfig,
    MarineGrid,
    MarinePredictionRequest,
    MarinePredictionResult,
)


def make_frames(value=1.0, steps=FORECAST_STEPS, ny=2, nx=2):
    return [[[value, value + 0.1], [value + 0.2, value + 0.3]] for _ in range(steps)]


class MarineGridTests(SimpleTestCase):
    def test_valid_grid_is_accepted(self):
        grid = MarineGrid(west=39.5, south=-4.2, east=39.8, north=-3.9, nx=3, ny=2)

        self.assertEqual(grid.crs, "EPSG:4326")
        self.assertEqual(grid.nx, 3)
        self.assertEqual(grid.ny, 2)

    def test_invalid_longitude_bounds_are_rejected(self):
        with self.assertRaises(ValueError):
            MarineGrid(west=40, south=-4.2, east=39.8, north=-3.9, nx=3, ny=2)

    def test_invalid_latitude_bounds_are_rejected(self):
        with self.assertRaises(ValueError):
            MarineGrid(west=39.5, south=-91, east=39.8, north=-3.9, nx=3, ny=2)

    def test_grid_dimensions_must_be_at_least_two(self):
        with self.assertRaises(ValueError):
            MarineGrid(west=39.5, south=-4.2, east=39.8, north=-3.9, nx=1, ny=2)

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
    def test_default_forecast_uses_six_hourly_steps(self):
        forecast = MarineForecastConfig(start_time="2026-10-20T10:00:00Z")

        self.assertEqual(forecast.steps, FORECAST_STEPS)
        self.assertEqual(forecast.step_seconds, FORECAST_STEP_SECONDS)
        self.assertEqual(forecast.steps, 6)
        self.assertEqual(forecast.step_seconds, 3600)

    def test_explicit_fixed_forecast_is_accepted(self):
        forecast = MarineForecastConfig(
            start_time="2026-10-20T10:00:00Z",
            steps=6,
            step_seconds=3600,
        )

        self.assertEqual(forecast.steps, 6)
        self.assertEqual(forecast.step_seconds, 3600)

    def test_timezone_is_required(self):
        with self.assertRaises(ValueError):
            MarineForecastConfig(start_time="2026-10-20T10:00:00")

    def test_invalid_timestamp_is_rejected(self):
        with self.assertRaises(ValueError):
            MarineForecastConfig(start_time="not-a-timestamp")

    def test_non_string_timestamp_is_rejected(self):
        with self.assertRaises(ValueError):
            MarineForecastConfig(start_time=None)

    def test_non_six_step_forecast_is_rejected(self):
        with self.assertRaises(ValueError):
            MarineForecastConfig(
                start_time="2026-10-20T10:00:00Z",
                steps=5,
            )

    def test_non_hourly_forecast_is_rejected(self):
        with self.assertRaises(ValueError):
            MarineForecastConfig(
                start_time="2026-10-20T10:00:00Z",
                step_seconds=900,
            )

    def test_fractional_steps_are_rejected(self):
        with self.assertRaises(ValueError):
            MarineForecastConfig(
                start_time="2026-10-20T10:00:00Z",
                steps=6.5,
            )

    def test_boolean_steps_are_rejected(self):
        with self.assertRaises(ValueError):
            MarineForecastConfig(
                start_time="2026-10-20T10:00:00Z",
                steps=True,
            )

    def test_fractional_step_interval_is_rejected(self):
        with self.assertRaises(ValueError):
            MarineForecastConfig(
                start_time="2026-10-20T10:00:00Z",
                step_seconds=3600.5,
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

        self.forecast = MarineForecastConfig(start_time="2026-10-20T10:00:00Z")

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
            "initial_current_u_ms": [
                [0.1, 0.2],
                [0.3, 0.4],
            ],
            "initial_current_v_ms": [
                [0.2, 0.3],
                [0.4, 0.5],
            ],
        }

    def test_valid_prediction_request_is_accepted(self):
        request = MarinePredictionRequest(**self.valid_data)

        self.assertEqual(request.grid.nx, 2)
        self.assertEqual(request.grid.ny, 2)
        self.assertEqual(request.forecast.steps, 6)

    def test_incorrect_bathymetry_shape_is_rejected(self):
        data = {
            **self.valid_data,
            "bathymetry_m": [[10.0, 11.0]],
        }

        with self.assertRaises(ValueError):
            MarinePredictionRequest(**data)

    def test_incorrect_water_level_shape_is_rejected(self):
        data = {
            **self.valid_data,
            "initial_water_level_m": [
                [1.0, 1.1],
                [1.2],
            ],
        }

        with self.assertRaises(ValueError):
            MarinePredictionRequest(**data)

    def test_incorrect_eastward_current_shape_is_rejected(self):
        data = {
            **self.valid_data,
            "initial_current_u_ms": [[0.1, 0.2]],
        }

        with self.assertRaises(ValueError):
            MarinePredictionRequest(**data)

    def test_incorrect_northward_current_shape_is_rejected(self):
        data = {
            **self.valid_data,
            "initial_current_v_ms": [
                [0.2, 0.3],
                [0.4],
            ],
        }

        with self.assertRaises(ValueError):
            MarinePredictionRequest(**data)

    def test_non_list_spatial_field_is_rejected(self):
        for field in (
            "bathymetry_m",
            "initial_water_level_m",
            "initial_current_u_ms",
            "initial_current_v_ms",
        ):
            with self.subTest(field=field):
                data = {**self.valid_data, field: None}

                with self.assertRaises(ValueError):
                    MarinePredictionRequest(**data)

    def test_nan_spatial_values_are_rejected(self):
        data = {
            **self.valid_data,
            "initial_current_u_ms": [
                [float("nan"), 0.2],
                [0.3, 0.4],
            ],
        }

        with self.assertRaises(ValueError):
            MarinePredictionRequest(**data)

    def test_infinite_spatial_values_are_rejected(self):
        data = {
            **self.valid_data,
            "initial_current_v_ms": [
                [0.2, 0.3],
                [0.4, float("inf")],
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

    def test_boolean_water_level_values_are_rejected(self):
        data = {
            **self.valid_data,
            "initial_water_level_m": [
                [1.0, False],
                [1.2, 1.3],
            ],
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
            "water_level_m": make_frames(1.0),
            "current_u_ms": make_frames(0.1),
            "current_v_ms": make_frames(0.2),
            "model_version": "fno-0.2.0",
        }

    def make_request(self, nx=2, ny=2):
        return MarinePredictionRequest(
            grid=MarineGrid(
                west=39.5,
                south=-4.2,
                east=39.8,
                north=-3.9,
                nx=nx,
                ny=ny,
            ),
            forecast=MarineForecastConfig(start_time="2026-10-20T10:00:00Z"),
            bathymetry_m=[[10.0] * nx for _ in range(ny)],
            initial_water_level_m=[[1.0] * nx for _ in range(ny)],
            initial_current_u_ms=[[0.1] * nx for _ in range(ny)],
            initial_current_v_ms=[[0.2] * nx for _ in range(ny)],
        )

    def test_valid_prediction_result_is_accepted(self):
        result = MarinePredictionResult(**self.valid_data)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.contract_version, CONTRACT_VERSION)
        self.assertEqual(result.model_version, "fno-0.2.0")

    def test_prediction_fields_must_have_matching_shapes(self):
        data = {
            **self.valid_data,
            "current_u_ms": make_frames(0.1, steps=5),
        }

        with self.assertRaises(ValueError):
            MarinePredictionResult(**data)

    def test_prediction_must_contain_six_forecast_frames(self):
        data = {
            **self.valid_data,
            "water_level_m": make_frames(1.0, steps=5),
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
        frames = make_frames(1.0)
        frames[0][0][0] = float("nan")

        data = {
            **self.valid_data,
            "water_level_m": frames,
        }

        with self.assertRaises(ValueError):
            MarinePredictionResult(**data)

    def test_infinite_current_values_are_rejected(self):
        frames = make_frames(0.1)
        frames[0][0][0] = float("inf")

        data = {
            **self.valid_data,
            "current_u_ms": frames,
        }

        with self.assertRaises(ValueError):
            MarinePredictionResult(**data)

    def test_boolean_current_values_are_rejected(self):
        frames = make_frames(0.2)
        frames[0][0][0] = True

        data = {
            **self.valid_data,
            "current_v_ms": frames,
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
        request = self.make_request(nx=3, ny=2)

        with self.assertRaises(ValueError):
            result.validate_against(request)

    def test_result_matching_request_dimensions_is_accepted(self):
        result = MarinePredictionResult(**self.valid_data)
        request = self.make_request()

        self.assertIsNone(result.validate_against(request))
