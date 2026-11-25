from dataclasses import dataclass, field

from port_twin_app.services.terminal_edi.parser import EDIFACTSegment


@dataclass(frozen=True)
class COPARNEvent:
    """Normalized container pre-advice extracted from a COPARN message."""

    event_type: str
    container_reference: str
    metadata: dict = field(default_factory=dict)


class COPARNAdapter:
    """Extract container pre-advice events from the supported COPARN subset."""

    MESSAGE_TYPE = "COPARN"
    EVENT_TYPE = "pre_advice"

    def parse(self, segments: list[EDIFACTSegment]) -> list[COPARNEvent]:
        if not segments:
            raise ValueError("COPARN message contains no segments.")

        message_type = self._get_message_type(segments)

        if message_type != self.MESSAGE_TYPE:
            raise ValueError(
                f"Expected COPARN message, received {message_type or 'unknown'}."
            )

        message_reference = self._get_message_reference(segments)

        shared_metadata = {
            "message_type": self.MESSAGE_TYPE,
            "message_reference": message_reference,
            "references": self._get_segment_data(segments, "RFF"),
            "dates": self._get_segment_data(segments, "DTM"),
            "locations": self._get_segment_data(segments, "LOC"),
            "parties": self._get_segment_data(segments, "NAD"),
            "transport": self._get_segment_data(segments, "TDT"),
        }

        events = []

        for segment in segments:
            if segment.tag != "EQD":
                continue

            equipment_qualifier = self._get_component(segment, 0)
            container_reference = self._get_component(segment, 1)

            # Only create events for identified containers.
            if equipment_qualifier != "CN" or not container_reference:
                continue

            metadata = {
                **shared_metadata,
                "equipment_qualifier": equipment_qualifier,
                "equipment_details": self._serialize_segment(segment),
            }

            events.append(
                COPARNEvent(
                    event_type=self.EVENT_TYPE,
                    container_reference=container_reference,
                    metadata=metadata,
                )
            )

        if not events:
            raise ValueError(
                "COPARN message contains no identifiable container EQD segments."
            )

        return events

    @staticmethod
    def _get_message_type(segments: list[EDIFACTSegment]) -> str | None:
        for segment in segments:
            if segment.tag == "UNH" and len(segment.elements) > 1:
                message_identifier = segment.elements[1]
                return message_identifier[0] if message_identifier else None

        return None

    @staticmethod
    def _get_message_reference(segments: list[EDIFACTSegment]) -> str:
        for segment in segments:
            if segment.tag == "BGM" and len(segment.elements) > 1:
                reference = segment.elements[1]
                return reference[0] if reference else ""

        return ""

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
    def _get_segment_data(
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
