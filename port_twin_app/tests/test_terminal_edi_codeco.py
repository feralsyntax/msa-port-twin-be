from django.test import SimpleTestCase

from port_twin_app.services.terminal_edi.adapters.codeco import CODECOAdapter
from port_twin_app.services.terminal_edi.parser import EDIFACTParser


class CODECOAdapterTests(SimpleTestCase):
    def setUp(self):
        self.parser = EDIFACTParser()
        self.adapter = CODECOAdapter()

    def test_extracts_gate_in_events(self):
        message = (
            "UNH+1+CODECO:D:95B:UN'"
            "BGM+34+GATE-IN-001+9'"
            "DTM+137:202611251000:203'"
            "TDT+20+VOYAGE-001'"
            "NAD+MS+TERMINAL-01'"
            "EQD+CN+MSCU1234567'"
            "DTM+7:202611251030:203'"
            "LOC+165+MOMBASA'"
            "RFF+BN:BOOKING-001'"
            "EQD+CN+MSCU7654321'"
            "DTM+7:202611251100:203'"
            "UNT+11+1'"
        )

        events = self.adapter.parse(self.parser.parse(message))

        self.assertEqual(len(events), 2)
        self.assertEqual(
            [event.event_type for event in events],
            ["gate_in", "gate_in"],
        )
        self.assertEqual(
            [event.container_reference for event in events],
            ["MSCU1234567", "MSCU7654321"],
        )

        self.assertEqual(
            events[0].metadata["message_reference"],
            "GATE-IN-001",
        )
        self.assertEqual(
            events[0].metadata["document_code"],
            "34",
        )
        self.assertEqual(
            events[0].metadata["dates"][-1]["elements"],
            [["7", "202611251030", "203"]],
        )
        self.assertEqual(
            events[0].metadata["locations"][0]["elements"],
            [["165"], ["MOMBASA"]],
        )
        self.assertEqual(
            events[1].metadata["dates"][-1]["elements"],
            [["7", "202611251100", "203"]],
        )

    def test_extracts_gate_out_event(self):
        message = (
            "UNH+1+CODECO:D:95B:UN'"
            "BGM+36+GATE-OUT-001+9'"
            "EQD+CN+MSCU1234567'"
            "DTM+7:202611251230:203'"
            "UNT+4+1'"
        )

        events = self.adapter.parse(self.parser.parse(message))

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "gate_out")
        self.assertEqual(events[0].container_reference, "MSCU1234567")

    def test_ignores_non_container_equipment(self):
        message = (
            "UNH+1+CODECO:D:95B:UN'"
            "BGM+34+GATE-IN-002+9'"
            "EQD+CH+TRAILER-001'"
            "EQD+CN+MSCU1234567'"
            "UNT+5+1'"
        )

        events = self.adapter.parse(self.parser.parse(message))

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].container_reference, "MSCU1234567")

    def test_rejects_non_codeco_message(self):
        message = "UNH+1+COPARN:D:95B:UN'BGM+34+MSG-001+9'EQD+CN+MSCU1234567'UNT+4+1'"

        with self.assertRaises(ValueError):
            self.adapter.parse(self.parser.parse(message))

    def test_rejects_unsupported_document_code(self):
        message = "UNH+1+CODECO:D:95B:UN'BGM+99+MSG-001+9'EQD+CN+MSCU1234567'UNT+4+1'"

        with self.assertRaises(ValueError):
            self.adapter.parse(self.parser.parse(message))

    def test_rejects_non_original_message_function(self):
        message = "UNH+1+CODECO:D:95B:UN'BGM+34+MSG-001+1'EQD+CN+MSCU1234567'UNT+4+1'"

        with self.assertRaises(ValueError):
            self.adapter.parse(self.parser.parse(message))

    def test_rejects_message_without_identifiable_containers(self):
        message = "UNH+1+CODECO:D:95B:UN'BGM+34+MSG-001+9'EQD+CH+TRAILER-001'UNT+4+1'"

        with self.assertRaises(ValueError):
            self.adapter.parse(self.parser.parse(message))

    def test_rejects_message_without_bgm(self):
        message = "UNH+1+CODECO:D:95B:UN'EQD+CN+MSCU1234567'UNT+3+1'"

        with self.assertRaises(ValueError):
            self.adapter.parse(self.parser.parse(message))
