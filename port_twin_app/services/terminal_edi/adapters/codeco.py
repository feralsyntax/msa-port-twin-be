from dataclasses import dataclass, field

from port_twin_app.services.terminal_edi.parser import EDIFACTSegment


@dataclass(frozen=True)
class CODECOEvent:
    """Normalized gate movement extracted from a CODECO message."""

    event_type: str
    container_reference: str
    metadata: dict = field(default_factory=dict)


class CODECOAdapter:
    """Extract gate-in and gate-out events from the supported CODECO subset."""

    MESSAGE_TYPE = "CODECO"

    DOCUMENT_EVENT_TYPES = {
        "34": "gate_in",
        "36": "gate_out",
    }

    ORIGINAL_MESSAGE_FUNCTION = "9"

    def parse(self, segments: list[EDIFACTSegment]) -> list[CODECOEvent]:
        if not segments:
            raise ValueError("CODECO message contains no segments.")

        message_type = self._get_message_type(segments)

        if message_type != self.MESSAGE_TYPE:
            raise ValueError(
                f"Expected CODECO message, received {message_type or 'unknown'}."
            )

        document_code, message_reference, message_function = self._get_message_details(
            segments
        )

        event_type = self.DOCUMENT_EVENT_TYPES.get(document_code)

        if event_type is None:
            raise ValueError(
                f"Unsupported CODECO document code: {document_code or 'missing'}."
            )

        if message_function != self.ORIGINAL_MESSAGE_FUNCTION:
            raise ValueError(
                "Only original CODECO messages are supported; "
                f"received message function {message_function or 'missing'}."
            )

        equipment_segments = [
            index for index, segment in enumerate(segments) if segment.tag == "EQD"
        ]

        if not equipment_segments:
            raise ValueError("CODECO message contains no EQD equipment segments.")

        first_equipment_index = equipment_segments[0]
        header_segments = segments[:first_equipment_index]

        shared_metadata = {
            "message_type": self.MESSAGE_TYPE,
            "message_reference": message_reference,
            "document_code": document_code,
            "message_function": message_function,
            "dates": self._serialize_segments(header_segments, "DTM"),
            "references": self._serialize_segments(header_segments, "RFF"),
            "transport": self._serialize_segments(header_segments, "TDT"),
            "parties": self._serialize_segments(header_segments, "NAD"),
        }

        events = []

        for position, start_index in enumerate(equipment_segments):
            if position + 1 < len(equipment_segments):
                end_index = equipment_segments[position + 1]
            else:
                end_index = len(segments)

            equipment_group = segments[start_index:end_index]
            equipment_segment = equipment_group[0]

            equipment_qualifier = self._get_component(
                equipment_segment,
                0,
            )
            container_reference = self._get_component(
                equipment_segment,
                1,
            )

            if equipment_qualifier != "CN" or not container_reference:
                continue

            metadata = {
                **shared_metadata,
                "equipment_details": self._serialize_segment(equipment_segment),
                "dates": (
                    shared_metadata["dates"]
                    + self._serialize_segments(equipment_group[1:], "DTM")
                ),
                "references": (
                    shared_metadata["references"]
                    + self._serialize_segments(equipment_group[1:], "RFF")
                ),
                "locations": self._serialize_segments(equipment_group[1:], "LOC"),
                "movement_details": self._serialize_segments(
                    equipment_group[1:],
                    "TMD",
                ),
            }

            events.append(
                CODECOEvent(
                    event_type=event_type,
                    container_reference=container_reference,
                    metadata=metadata,
                )
            )

        if not events:
            raise ValueError(
                "CODECO message contains no identifiable container EQD segments."
            )

        return events

    @staticmethod
    def _get_message_type(segments: list[EDIFACTSegment]) -> str | None:
        for segment in segments:
            if segment.tag == "UNH" and len(segment.elements) > 1:
                identifier = segment.elements[1]
                return identifier[0] if identifier else None

        return None

    @classmethod
    def _get_message_details(
        cls,
        segments: list[EDIFACTSegment],
    ) -> tuple[str, str, str]:
        for segment in segments:
            if segment.tag != "BGM":
                continue

            document_code = cls._get_component(segment, 0)
            message_reference = cls._get_component(segment, 1)
            message_function = cls._get_component(segment, 2)

            return document_code, message_reference, message_function

        raise ValueError("CODECO message is missing its BGM segment.")

    @staticmethod
    def _get_component(
        segment: EDIFACTSegment,
        element_index: int,
    ) -> str:
        if element_index >= len(segment.elements):
            return ""

        components = segment.elements[element_index]
        return components[0] if components else ""

    @classmethod
    def _serialize_segments(
        cls,
        segments: list[EDIFACTSegment],
        tag: str,
    ) -> list[dict]:
        return [
            cls._serialize_segment(segment)
            for segment in segments
            if segment.tag == tag
        ]

    @staticmethod
    def _serialize_segment(segment: EDIFACTSegment) -> dict:
        return {
            "tag": segment.tag,
            "elements": [list(components) for components in segment.elements],
        }
