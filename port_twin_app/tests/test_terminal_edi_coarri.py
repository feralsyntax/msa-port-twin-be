from django.test import SimpleTestCase

from port_twin_app.services.terminal_edi.adapters.coarri import COARRIAdapter
from port_twin_app.services.terminal_edi.parser import EDIFACTParser


class COARRIAdapterTests(SimpleTestCase):
    def setUp(self):
        self.parser = EDIFACTParser()
        self.adapter = COARRIAdapter()

    def test_extracts_loading_events(self):
        message = (
            "UNH+1+COARRI:D:95B:UN'"
            "BGM+270+LOAD-001+9'"
            "TDT+20+4321+1++HLC:172:20'"
            "LOC+9+MOMBASA:139:6+BERTH-01:BER:ZZZ'"
            "DTM+132:202611251000:203'"
            "NAD+CA+HLC:160:20'"
            "EQD+CN+MSCU1234567+2200:102:5+++5'"
            "RFF+BN:BOOKING-001'"
            "TMD+11'"
            "DTM+203:202611251030:203'"
            "LOC+11+SGSIN:139:6'"
            "LOC+147+0340210::5'"
            "MEA+WT+G+KGM:13450'"
            "EQD+CN+MSCU7654321+2200:102:5+++5'"
            "RFF+BN:BOOKING-002'"
            "DTM+203:202611251100:203'"
            "UNT+16+1'"
        )

        events = self.adapter.parse(self.parser.parse(message))

        self.assertEqual(len(events), 2)
        self.assertEqual(
            [event.event_type for event in events],
            ["loading", "loading"],
        )
        self.assertEqual(
            [event.container_reference for event in events],
            ["MSCU1234567", "MSCU7654321"],
        )

        self.assertEqual(
            events[0].metadata["message_reference"],
            "LOAD-001",
        )
        self.assertEqual(
            events[0].metadata["transport"][0]["elements"][1],
            ["4321"],
        )
        self.assertEqual(
            events[0].metadata["container_dates"][0]["elements"],
            [["203", "202611251030", "203"]],
        )
        self.assertEqual(
            events[0].metadata["container_references"][0]["elements"],
            [["BN", "BOOKING-001"]],
        )
        self.assertEqual(
            events[1].metadata["container_references"][0]["elements"],
            [["BN", "BOOKING-002"]],
        )

    def test_extracts_discharge_events(self):
        message = (
            "UNH+1+COARRI:D:95B:UN'"
            "BGM+98+DISCHARGE-001+9'"
            "TDT+20+4321+1++HLC:172:20'"
            "LOC+11+MOMBASA:139:6'"
            "EQD+CN+HLCU2603437+2250:102:5+++5'"
            "RFF+BN:BOOKING-003'"
            "DTM+203:202611251230:203'"
            "LOC+9+SGSIN:139:6'"
            "UNT+9+1'"
        )

        events = self.adapter.parse(self.parser.parse(message))

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "discharge")
        self.assertEqual(events[0].container_reference, "HLCU2603437")

    def test_ignores_non_container_equipment(self):
        message = (
            "UNH+1+COARRI:D:95B:UN'"
            "BGM+98+DISCHARGE-002+9'"
            "EQD+BB+++++5'"
            "EQD+CN+MSCU1234567'"
            "UNT+5+1'"
        )

        events = self.adapter.parse(self.parser.parse(message))

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].container_reference, "MSCU1234567")

    def test_keeps_container_details_separate(self):
        message = (
            "UNH+1+COARRI:D:95B:UN'"
            "BGM+98+DISCHARGE-003+9'"
            "EQD+CN+MSCU1234567'"
            "RFF+BN:BOOKING-001'"
            "EQD+CN+MSCU7654321'"
            "RFF+BN:BOOKING-002'"
            "UNT+6+1'"
        )

        events = self.adapter.parse(self.parser.parse(message))

        self.assertEqual(
            events[0].metadata["container_references"][0]["elements"],
            [["BN", "BOOKING-001"]],
        )
        self.assertEqual(
            events[1].metadata["container_references"][0]["elements"],
            [["BN", "BOOKING-002"]],
        )

    def test_rejects_non_coarri_message(self):
        message = "UNH+1+CODECO:D:95B:UN'BGM+98+MSG-001+9'EQD+CN+MSCU1234567'UNT+4+1'"

        with self.assertRaises(ValueError):
            self.adapter.parse(self.parser.parse(message))

    def test_rejects_unsupported_document_code(self):
        message = "UNH+1+COARRI:D:95B:UN'BGM+999+MSG-001+9'EQD+CN+MSCU1234567'UNT+4+1'"

        with self.assertRaises(ValueError):
            self.adapter.parse(self.parser.parse(message))

    def test_rejects_non_original_message_function(self):
        message = "UNH+1+COARRI:D:95B:UN'BGM+98+MSG-001+1'EQD+CN+MSCU1234567'UNT+4+1'"

        with self.assertRaises(ValueError):
            self.adapter.parse(self.parser.parse(message))

    def test_rejects_message_without_identifiable_containers(self):
        message = "UNH+1+COARRI:D:95B:UN'BGM+98+MSG-001+9'EQD+BB+++++5'UNT+4+1'"

        with self.assertRaises(ValueError):
            self.adapter.parse(self.parser.parse(message))

    def test_rejects_message_without_bgm(self):
        message = "UNH+1+COARRI:D:95B:UN'EQD+CN+MSCU1234567'UNT+3+1'"

        with self.assertRaises(ValueError):
            self.adapter.parse(self.parser.parse(message))
