from django.test import TestCase

from port_twin_app.models import TerminalEDIMessage, TerminalOperationalEvent
from port_twin_app.services.terminal_edi.ingestion import (
    TerminalEDIIngestionService,
)


class TerminalEDIIngestionTests(TestCase):
    def setUp(self):
        self.service = TerminalEDIIngestionService(source="test-terminal")

    def test_ingests_coparn_message(self):
        message = "UNH+1+COPARN:D:95B:UN'BGM+11+PRE-001+9'EQD+CN+MSCU1234567'UNT+4+1'"

        result = self.service.ingest(message)

        self.assertEqual(result["status"], "processed")
        self.assertFalse(result["duplicate"])
        self.assertEqual(result["events_created"], 1)

        stored_message = TerminalEDIMessage.objects.get()
        event = TerminalOperationalEvent.objects.get()

        self.assertEqual(stored_message.message_type, "COPARN")
        self.assertEqual(stored_message.message_reference, "PRE-001")
        self.assertEqual(event.event_type, "pre_advice")
        self.assertEqual(event.container_reference, "MSCU1234567")

    def test_ingests_codeco_message(self):
        message = (
            "UNH+1+CODECO:D:95B:UN'BGM+34+GATE-IN-001+9'EQD+CN+MSCU1234567'UNT+4+1'"
        )

        result = self.service.ingest(message)

        self.assertEqual(result["events_created"], 1)
        self.assertEqual(
            TerminalOperationalEvent.objects.get().event_type,
            "gate_in",
        )

    def test_ingests_coarri_message(self):
        message = (
            "UNH+1+COARRI:D:95B:UN'BGM+98+DISCHARGE-001+9'EQD+CN+MSCU1234567'UNT+4+1'"
        )

        result = self.service.ingest(message)

        self.assertEqual(result["events_created"], 1)
        self.assertEqual(
            TerminalOperationalEvent.objects.get().event_type,
            "discharge",
        )

    def test_does_not_ingest_duplicate_message_twice(self):
        message = "UNH+1+COPARN:D:95B:UN'BGM+11+PRE-002+9'EQD+CN+MSCU1234567'UNT+4+1'"

        first_result = self.service.ingest(message)
        second_result = self.service.ingest(message)

        self.assertFalse(first_result["duplicate"])
        self.assertTrue(second_result["duplicate"])
        self.assertEqual(second_result["events_created"], 0)
        self.assertEqual(TerminalEDIMessage.objects.count(), 1)
        self.assertEqual(TerminalOperationalEvent.objects.count(), 1)

    def test_same_message_from_different_sources_is_not_duplicate(self):
        message = "UNH+1+COPARN:D:95B:UN'BGM+11+PRE-003+9'EQD+CN+MSCU1234567'UNT+4+1'"

        other_service = TerminalEDIIngestionService(source="another-terminal")

        self.service.ingest(message)
        result = other_service.ingest(message)

        self.assertFalse(result["duplicate"])
        self.assertEqual(TerminalEDIMessage.objects.count(), 2)
        self.assertEqual(TerminalOperationalEvent.objects.count(), 2)

    def test_preserves_synthetic_provenance(self):
        message = "UNH+1+COPARN:D:95B:UN'BGM+11+PRE-004+9'EQD+CN+MSCU1234567'UNT+4+1'"

        self.service.ingest(message, is_synthetic=True)

        stored_message = TerminalEDIMessage.objects.get()
        self.assertTrue(stored_message.is_synthetic)

    def test_records_adapter_failure(self):
        message = (
            "UNH+1+CODECO:D:95B:UN'BGM+99+INVALID-001+9'EQD+CN+MSCU1234567'UNT+4+1'"
        )

        with self.assertRaises(ValueError):
            self.service.ingest(message)

        stored_message = TerminalEDIMessage.objects.get()

        self.assertEqual(stored_message.status, "failed")
        self.assertTrue(stored_message.error_message)
        self.assertEqual(TerminalOperationalEvent.objects.count(), 0)

    def test_rejects_unsupported_message_type(self):
        message = "UNH+1+INVOIC:D:95B:UN'BGM+380+INVOICE-001+9'UNT+3+1'"

        with self.assertRaises(ValueError):
            self.service.ingest(message)

        self.assertEqual(TerminalEDIMessage.objects.count(), 0)

    def test_rejects_empty_source(self):
        with self.assertRaises(ValueError):
            TerminalEDIIngestionService(source=" ")

    def test_rejects_empty_message(self):
        with self.assertRaises(ValueError):
            self.service.ingest(" ")

        self.assertEqual(TerminalEDIMessage.objects.count(), 0)
