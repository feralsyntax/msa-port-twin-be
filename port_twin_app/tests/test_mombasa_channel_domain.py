from django.test import SimpleTestCase

from port_twin_app.services.marine_data.domains.mombasa_channel import (
    APPROACHES,
    GEOGRAPHIC_CRS,
    PORT_OPERATIONS,
    PROJECTED_CRS,
    GeographicBounds,
    get_zone,
    validate_domain_zones,
)


class GeographicBoundsTests(SimpleTestCase):
    def test_rejects_invalid_longitude_order(self):
        with self.assertRaises(ValueError):
            GeographicBounds(
                west=40.0,
                south=-4.0,
                east=39.0,
                north=-3.0,
            )

    def test_rejects_invalid_latitude_order(self):
        with self.assertRaises(ValueError):
            GeographicBounds(
                west=39.0,
                south=-3.0,
                east=40.0,
                north=-4.0,
            )

    def test_parent_contains_child(self):
        self.assertTrue(APPROACHES.bounds.contains(PORT_OPERATIONS.bounds))


class MombasaChannelDomainTests(SimpleTestCase):
    def test_uses_expected_coordinate_reference_systems(self):
        self.assertEqual(GEOGRAPHIC_CRS, "EPSG:4326")
        self.assertEqual(PROJECTED_CRS, "EPSG:32737")

    def test_port_operations_is_finer_than_approaches(self):
        self.assertLess(
            PORT_OPERATIONS.target_resolution_m,
            APPROACHES.target_resolution_m,
        )

    def test_domain_zones_are_valid(self):
        validate_domain_zones()

    def test_get_zone_returns_configured_zone(self):
        self.assertEqual(get_zone("approaches"), APPROACHES)
        self.assertEqual(get_zone("port_operations"), PORT_OPERATIONS)

    def test_unknown_zone_is_rejected(self):
        with self.assertRaises(ValueError):
            get_zone("unknown")
