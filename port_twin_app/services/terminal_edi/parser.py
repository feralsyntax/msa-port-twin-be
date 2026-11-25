from dataclasses import dataclass


@dataclass(frozen=True)
class EDIFACTSegment:
    tag: str
    elements: tuple[tuple[str, ...], ...]


class EDIFACTParser:
    """Parse the structural subset of UN/EDIFACT used by terminal adapters."""

    DEFAULT_COMPONENT_SEPARATOR = ":"
    DEFAULT_ELEMENT_SEPARATOR = "+"
    DEFAULT_DECIMAL_MARK = "."
    DEFAULT_RELEASE_CHARACTER = "?"
    DEFAULT_SEGMENT_TERMINATOR = "'"

    def parse(self, raw_message: str) -> list[EDIFACTSegment]:
        if not isinstance(raw_message, str) or not raw_message.strip():
            raise ValueError("EDI message cannot be empty.")

        message = raw_message.strip().lstrip("\ufeff")

        delimiters = self._get_delimiters(message)

        if message.startswith("UNA"):
            message = message[9:]

        segments = self._split_escaped(
            message,
            delimiters["segment"],
            delimiters["release"],
        )

        parsed = []

        for raw_segment in segments:
            raw_segment = raw_segment.strip()

            if not raw_segment:
                continue

            elements = self._split_escaped(
                raw_segment,
                delimiters["element"],
                delimiters["release"],
            )

            tag = elements[0].strip()

            if not tag:
                raise ValueError("EDI segment has an empty tag.")

            parsed_elements = []

            for element in elements[1:]:
                components = self._split_escaped(
                    element,
                    delimiters["component"],
                    delimiters["release"],
                )

                parsed_elements.append(
                    tuple(
                        self._unescape(component, delimiters["release"])
                        for component in components
                    )
                )

            parsed.append(
                EDIFACTSegment(
                    tag=tag,
                    elements=tuple(parsed_elements),
                )
            )

        if not parsed:
            raise ValueError("EDI message contains no segments.")

        return parsed

    def _get_delimiters(self, message: str) -> dict[str, str]:
        if message.startswith("UNA"):
            if len(message) < 9:
                raise ValueError("Invalid UNA service string advice.")

            return {
                "component": message[3],
                "element": message[4],
                "decimal": message[5],
                "release": message[6],
                "segment": message[8],
            }

        return {
            "component": self.DEFAULT_COMPONENT_SEPARATOR,
            "element": self.DEFAULT_ELEMENT_SEPARATOR,
            "decimal": self.DEFAULT_DECIMAL_MARK,
            "release": self.DEFAULT_RELEASE_CHARACTER,
            "segment": self.DEFAULT_SEGMENT_TERMINATOR,
        }

    @staticmethod
    def _split_escaped(value: str, separator: str, release: str) -> list[str]:
        parts = []
        current = []
        escaped = False

        for character in value:
            if escaped:
                # Preserve the release character for subsequent parsing stages.
                current.extend((release, character))
                escaped = False
            elif character == release:
                escaped = True
            elif character == separator:
                parts.append("".join(current))
                current = []
            else:
                current.append(character)

        if escaped:
            raise ValueError("EDI message ends with an incomplete release character.")

        parts.append("".join(current))
        return parts

    @staticmethod
    def _unescape(value: str, release: str) -> str:
        result = []
        escaped = False

        for character in value:
            if escaped:
                result.append(character)
                escaped = False
            elif character == release:
                escaped = True
            else:
                result.append(character)

        if escaped:
            raise ValueError("EDI element ends with an incomplete release character.")

        return "".join(result)
