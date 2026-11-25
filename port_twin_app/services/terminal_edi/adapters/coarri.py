from dataclasses import dataclass, field

from port_twin_app.services.terminal_edi.parser import EDIFACTSegment


@dataclass(frozen=True)
class COARRIEvent:
    """Normalized vessel container-handling event extracted from COARRI."""

    event_type: str
    container_reference: str
    metadata: dict = field(default_factory=dict)


class COARRIAdapter:
    """Extract container discharge/loading events from the D.95B subset."""

    MESSAGE_TYPE = "COARRI"

    DOCUMENT_EVENT_TYPES = {
        "98": "discharge",
        "270": "loading",
    }

    ORIGINAL_MESSAGE_FUNCTION = "9"

    def parse(self, segments: list[EDIFACTSegment]) -> list[COARRIEvent]:
        if not segments:
            raise ValueError("COARRI message contains no segments.")

        message_type = self._get_message_type(segments)

        if message_type != self.MESSAGE_TYPE:
            raise ValueError(
                f"Expected COARRI message, received {message_type or 'unknown'}."
            )

        document_code, message_reference, message_function = self._get_message_details(
            segments
        )

        event_type = self.DOCUMENT_EVENT_TYPES.get(document_code)

        if event_type is None:
            raise ValueError(
                f"Unsupported COARRI document code: {document_code or 'missing'}."
            )

        if message_function != self.ORIGINAL_MESSAGE_FUNCTION:
            raise ValueError(
                "Only original COARRI messages are supported; "
                f"received message function {message_function or 'missing'}."
            )

        equipment_indexes = [
            index for index, segment in enumerate(segments) if segment.tag == "EQD"
        ]

        if not equipment_indexes:
            raise ValueError("COARRI message contains no EQD equipment segments.")

        first_equipment_index = equipment_indexes[0]
        header_segments = segments[:first_equipment_index]

        shared_metadata = {
            "message_type": self.MESSAGE_TYPE,
            "message_reference": message_reference,
            "document_code": document_code,
            "message_function": message_function,
            "transport": self._serialize_segments(header_segments, "TDT"),
            "header_dates": self._serialize_segments(header_segments, "DTM"),
            "header_locations": self._serialize_segments(header_segments, "LOC"),
            "header_references": self._serialize_segments(header_segments, "RFF"),
            "parties": self._serialize_segments(header_segments, "NAD"),
        }

        events = []

        for position, start_index in enumerate(equipment_indexes):
            end_index = (
                equipment_indexes[position + 1]
                if position + 1 < len(equipment_indexes)
                else len(segments)
            )

            container_group = segments[start_index:end_index]
            equipment_segment = container_group[0]

            equipment_qualifier = self._get_component(equipment_segment, 0)
            container_reference = self._get_component(equipment_segment, 1)

            # Only create events for identified containers, not other equipment.
            if equipment_qualifier != "CN" or not container_reference:
                continue

            group_segments = container_group[1:]

            metadata = {
                **shared_metadata,
                "equipment_details": self._serialize_segment(equipment_segment),
                "container_references": self._serialize_segments(
                    group_segments,
                    "RFF",
                ),
                "container_dates": self._serialize_segments(
                    group_segments,
                    "DTM",
                ),
                "container_locations": self._serialize_segments(
                    group_segments,
                    "LOC",
                ),
                "measurements": self._serialize_segments(
                    group_segments,
                    "MEA",
                ),
                "movement_details": self._serialize_segments(
                    group_segments,
                    "TMD",
                ),
                "container_segments": [
                    self._serialize_segment(segment) for segment in group_segments
                ],
            }

            events.append(
                COARRIEvent(
                    event_type=event_type,
                    container_reference=container_reference,
                    metadata=metadata,
                )
            )

        if not events:
            raise ValueError(
                "COARRI message contains no identifiable container EQD segments."
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

        raise ValueError("COARRI message is missing its BGM segment.")

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
