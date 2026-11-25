from django.test import SimpleTestCase

from port_twin_app.services.terminal_edi.adapters.coparn import COPARNAdapter
from port_twin_app.services.terminal_edi.parser import EDIFACTParser


class COPARNAdapterTests(SimpleTestCase):
    def setUp(self):
        self.parser = EDIFACTParser()
        self.adapter = COPARNAdapter()

    def test_extracts_container_pre_advice_events(self):
        message = (
            "UNH+1+COPARN:D:95B:UN'"
            "BGM+11+PRE-001+9'"
            "DTM+137:20261125:102'"
            "RFF+BN:BOOKING-001'"
            "LOC+9+MOMBASA'"
            "TDT+20+VOYAGE-001'"
            "EQD+CN+MSCU1234567'"
            "EQD+CN+MSCU7654321'"
            "UNT+9+1'"
        )

        segments = self.parser.parse(message)
        events = self.adapter.parse(segments)

        self.assertEqual(len(events), 2)

        self.assertEqual(
            [event.event_type for event in events],
            ["pre_advice", "pre_advice"],
        )

        self.assertEqual(
            [event.container_reference for event in events],
            ["MSCU1234567", "MSCU7654321"],
        )

        self.assertEqual(
            events[0].metadata["message_reference"],
            "PRE-001",
        )

        self.assertEqual(
            events[0].metadata["references"][0]["elements"],
            [["BN", "BOOKING-001"]],
        )

    def test_ignores_non_container_equipment(self):
        message = (
            "UNH+1+COPARN:D:95B:UN'"
            "BGM+11+PRE-002+9'"
            "EQD+CH+TRAILER-001'"
            "EQD+CN+MSCU1234567'"
            "UNT+4+1'"
        )

        events = self.adapter.parse(self.parser.parse(message))

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].container_reference, "MSCU1234567")

    def test_rejects_non_coparn_message(self):
        message = "UNH+1+CODECO:D:95B:UN'BGM+36+MSG-001+9'EQD+CN+MSCU1234567'UNT+4+1'"

        with self.assertRaises(ValueError):
            self.adapter.parse(self.parser.parse(message))

    def test_rejects_message_without_identifiable_containers(self):
        message = "UNH+1+COPARN:D:95B:UN'BGM+11+PRE-003+9'EQD+CH+TRAILER-001'UNT+4+1'"

        with self.assertRaises(ValueError):
            self.adapter.parse(self.parser.parse(message))

    def test_rejects_message_without_unh(self):
        message = "BGM+11+PRE-004+9'EQD+CN+MSCU1234567'"

        with self.assertRaises(ValueError):
            self.adapter.parse(self.parser.parse(message))
