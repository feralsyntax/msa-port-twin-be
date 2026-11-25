import hashlib
import json

from django.db import transaction
from django.utils import timezone

from port_twin_app.models import TerminalEDIMessage, TerminalOperationalEvent
from port_twin_app.services.terminal_edi.adapters.coarri import COARRIAdapter
from port_twin_app.services.terminal_edi.adapters.codeco import CODECOAdapter
from port_twin_app.services.terminal_edi.adapters.coparn import COPARNAdapter
from port_twin_app.services.terminal_edi.parser import EDIFACTParser


class TerminalEDIIngestionService:
    """Persist terminal EDI messages and their normalized operational events."""

    ADAPTERS = {
        "COPARN": COPARNAdapter,
        "CODECO": CODECOAdapter,
        "COARRI": COARRIAdapter,
    }

    def __init__(self, source: str):
        if not isinstance(source, str) or not source.strip():
            raise ValueError("EDI source cannot be empty.")

        self.source = source.strip()
        self.parser = EDIFACTParser()

    def ingest(
        self,
        raw_message: str,
        *,
        is_synthetic: bool = False,
        received_at=None,
    ) -> dict:
        """Ingest one EDI message and return its processing result."""
        if not isinstance(raw_message, str) or not raw_message.strip():
            raise ValueError("EDI message cannot be empty.")

        segments = self.parser.parse(raw_message)
        message_type = self._get_message_type(segments)

        adapter_class = self.ADAPTERS.get(message_type)

        if adapter_class is None:
            raise ValueError(
                f"Unsupported terminal EDI message type: {message_type or 'unknown'}."
            )

        fingerprint = hashlib.sha256(raw_message.encode("utf-8")).hexdigest()

        message_reference = self._get_message_reference(segments)

        edi_message, created = TerminalEDIMessage.objects.get_or_create(
            source=self.source,
            fingerprint=fingerprint,
            defaults={
                "message_type": message_type,
                "message_reference": message_reference,
                "raw_message": raw_message,
                "is_synthetic": is_synthetic,
                "received_at": received_at or timezone.now(),
            },
        )

        if not created:
            return {
                "message_id": str(edi_message.pk),
                "status": edi_message.status,
                "duplicate": True,
                "events_created": 0,
            }

        try:
            normalized_events = adapter_class().parse(segments)

            with transaction.atomic():
                event_records = []

                for index, event in enumerate(normalized_events):
                    event_payload = {
                        "index": index,
                        "event_type": event.event_type,
                        "container_reference": event.container_reference,
                        "metadata": event.metadata,
                    }

                    event_fingerprint = self._event_fingerprint(event_payload)

                    event_records.append(
                        TerminalOperationalEvent(
                            edi_message=edi_message,
                            event_fingerprint=event_fingerprint,
                            event_type=event.event_type,
                            container_reference=event.container_reference,
                            metadata=event.metadata,
                        )
                    )

                TerminalOperationalEvent.objects.bulk_create(event_records)

                edi_message.status = TerminalEDIMessage.ProcessingStatus.PROCESSED
                edi_message.processed_at = timezone.now()
                edi_message.error_message = ""
                edi_message.save(
                    update_fields=["status", "processed_at", "error_message"]
                )

            return {
                "message_id": str(edi_message.pk),
                "status": edi_message.status,
                "duplicate": False,
                "events_created": len(event_records),
            }

        except Exception as exc:
            edi_message.status = TerminalEDIMessage.ProcessingStatus.FAILED
            edi_message.error_message = str(exc)
            edi_message.processed_at = timezone.now()
            edi_message.save(update_fields=["status", "error_message", "processed_at"])
            raise

    @staticmethod
    def _get_message_type(segments) -> str | None:
        for segment in segments:
            if segment.tag == "UNH" and len(segment.elements) > 1:
                identifier = segment.elements[1]
                return identifier[0] if identifier else None

        return None

    @staticmethod
    def _get_message_reference(segments) -> str:
        for segment in segments:
            if segment.tag == "BGM" and len(segment.elements) > 1:
                reference = segment.elements[1]
                return reference[0] if reference else ""

        return ""

    @staticmethod
    def _event_fingerprint(payload: dict) -> str:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        return hashlib.sha256(encoded).hexdigest()
