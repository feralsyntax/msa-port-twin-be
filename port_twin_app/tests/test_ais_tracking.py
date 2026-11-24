from django.test import SimpleTestCase

from port_twin_app.services.ais.tracking import (
    VesselZone,
    classify_position,
)


class VesselTrackingTests(SimpleTestCase):
    def test_classifies_port_operations_position(self):
        self.assertEqual(
            classify_position(39.65, -4.07),
            VesselZone.PORT_OPERATIONS,
        )

    def test_classifies_approaches_position(self):
        self.assertEqual(
            classify_position(39.75, -4.15),
            VesselZone.APPROACHES,
        )

    def test_classifies_position_outside_approaches(self):
        self.assertEqual(
            classify_position(40.0, -4.05),
            VesselZone.OUTSIDE_APPROACHES,
        )

    def test_returns_unknown_for_missing_coordinates(self):
        self.assertEqual(
            classify_position(None, None),
            VesselZone.UNKNOWN,
        )

    def test_returns_unknown_for_invalid_coordinates(self):
        self.assertEqual(
            classify_position(181, -4.05),
            VesselZone.UNKNOWN,
        )
