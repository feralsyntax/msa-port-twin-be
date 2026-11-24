from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from django.test import SimpleTestCase

from port_twin_app.services.ais.decoder import AISDecoder


class AISDecoderTests(SimpleTestCase):
    def setUp(self):
        self.decoder = AISDecoder()
        self.received_at = datetime(2026, 11, 21, 7, 0, tzinfo=UTC)

    def message(self, **overrides):
        values = {
            "msg_type": 1,
            "mmsi": 123456789,
            "lon": 39.7,
            "lat": -4.05,
            "speed": 12.5,
            "course": 180.0,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_decodes_valid_position(self):
        result = self.decoder.decode_message(
            self.message(),
            self.received_at,
        )

        self.assertEqual(result.mmsi, "123456789")
        self.assertEqual(result.longitude, 39.7)
        self.assertEqual(result.latitude, -4.05)
        self.assertEqual(result.speed_over_ground, 12.5)
        self.assertTrue(result.has_position)

    def test_rejects_invalid_mmsi(self):
        result = self.decoder.decode_message(
            self.message(mmsi=12345),
            self.received_at,
        )
        self.assertIsNone(result)

    def test_retains_valid_position_outside_approaches(self):
        result = self.decoder.decode_message(
            self.message(lon=40.0, lat=-4.05),
            self.received_at,
        )

        self.assertIsNotNone(result)
        self.assertTrue(result.has_position)
        self.assertEqual(result.longitude, 40.0)
        self.assertEqual(result.latitude, -4.05)

    def test_rejects_invalid_coordinates(self):
        result = self.decoder.decode_message(
            self.message(lon=181, lat=91),
            self.received_at,
        )
        self.assertFalse(result.has_position)

    def test_rejects_non_finite_coordinates(self):
        result = self.decoder.decode_message(
            self.message(lon=float("nan")),
            self.received_at,
        )
        self.assertFalse(result.has_position)

    def test_handles_unavailable_speed_and_course(self):
        result = self.decoder.decode_message(
            self.message(speed=102.3, course=360),
            self.received_at,
        )
        self.assertIsNone(result.speed_over_ground)
        self.assertIsNone(result.course_over_ground)

    def test_decodes_type_5_static_data(self):
        result = self.decoder.decode_message(
            self.message(
                msg_type=5,
                shipname="MV TEST SHIP@@@@",
                imo=9876543,
                ship_type=80,
                to_bow=100,
                to_stern=50,
                to_port=15,
                to_starboard=15,
                draught=8.2,
            ),
            self.received_at,
        )

        self.assertEqual(result.name, "MV TEST SHIP")
        self.assertEqual(result.imo_number, "9876543")
        self.assertEqual(result.vessel_type, "tanker")
        self.assertEqual(result.length_overall, 150)
        self.assertEqual(result.width, 30)
        self.assertEqual(result.draft, 8.2)
        self.assertFalse(result.has_position)

    def test_maps_type_70_to_general_cargo(self):
        result = self.decoder.decode_message(
            self.message(msg_type=5, ship_type=70),
            self.received_at,
        )
        self.assertEqual(result.vessel_type, "general_cargo")

    def test_pairs_type_24_parts(self):
        part_a = self.message(
            msg_type=24,
            partno=0,
            shipname="MV TEST@@@@",
        )
        part_b = self.message(
            msg_type=24,
            partno=1,
            ship_type=70,
            to_bow=80,
            to_stern=20,
            to_port=10,
            to_starboard=10,
        )

        first = self.decoder.decode_message(part_a, self.received_at)
        second = self.decoder.decode_message(
            part_b,
            self.received_at + timedelta(seconds=1),
        )

        self.assertIsNone(first)
        self.assertEqual(second.name, "MV TEST")
        self.assertEqual(second.vessel_type, "general_cargo")
        self.assertEqual(second.length_overall, 100)
        self.assertEqual(second.width, 20)

    def test_discards_expired_type_24_part(self):
        part_a = self.message(
            msg_type=24,
            partno=0,
            shipname="MV TEST",
        )
        part_b = self.message(
            msg_type=24,
            partno=1,
            ship_type=70,
        )

        self.decoder.decode_message(part_a, self.received_at)

        result = self.decoder.decode_message(
            part_b,
            self.received_at + timedelta(minutes=11),
        )

        self.assertIsNone(result)
