from django.test import SimpleTestCase

from port_twin_app.services.marine_data.sources.bathymetry import (
    BathymetryRequest,
)


class BathymetryRequestTests(SimpleTestCase):
    def test_valid_request(self):
        request = BathymetryRequest(
            west=39.5,
            south=-4.3,
            east=40.0,
            north=-3.7,
        )

        self.assertEqual(request.west, 39.5)
        self.assertEqual(request.south, -4.3)
        self.assertEqual(request.east, 40.0)
        self.assertEqual(request.north, -3.7)

    def test_rejects_invalid_longitude(self):
        invalid_values = [
            {"west": -181},
            {"east": 181},
        ]

        for override in invalid_values:
            with self.subTest(override=override):
                coordinates = {
                    "west": 39.5,
                    "south": -4.3,
                    "east": 40.0,
                    "north": -3.7,
                }
                coordinates.update(override)

                with self.assertRaises(ValueError):
                    BathymetryRequest(**coordinates)

    def test_rejects_invalid_latitude(self):
        invalid_values = [
            {"south": -91},
            {"north": 91},
        ]

        for override in invalid_values:
            with self.subTest(override=override):
                coordinates = {
                    "west": 39.5,
                    "south": -4.3,
                    "east": 40.0,
                    "north": -3.7,
                }
                coordinates.update(override)

                with self.assertRaises(ValueError):
                    BathymetryRequest(**coordinates)

    def test_rejects_reversed_longitude_bounds(self):
        with self.assertRaises(ValueError):
            BathymetryRequest(
                west=40.0,
                south=-4.3,
                east=39.5,
                north=-3.7,
            )

    def test_rejects_reversed_latitude_bounds(self):
        with self.assertRaises(ValueError):
            BathymetryRequest(
                west=39.5,
                south=-3.7,
                east=40.0,
                north=-4.3,
            )

    def test_rejects_non_finite_coordinates(self):
        invalid_values = [
            {"west": float("nan")},
            {"east": float("inf")},
            {"south": float("-inf")},
        ]

        for override in invalid_values:
            with self.subTest(override=override):
                coordinates = {
                    "west": 39.5,
                    "south": -4.3,
                    "east": 40.0,
                    "north": -3.7,
                }
                coordinates.update(override)

                with self.assertRaises(ValueError):
                    BathymetryRequest(**coordinates)

    def test_rejects_boolean_coordinates(self):
        with self.assertRaises(ValueError):
            BathymetryRequest(
                west=True,
                south=-4.3,
                east=40.0,
                north=-3.7,
            )
