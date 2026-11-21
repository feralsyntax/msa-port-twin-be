"""Decode and normalise AIS messages for the Mombasa Port Twin."""

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import isfinite

from pyais.stream import IterMessages

from port_twin_app.services.marine_data.domains.mombasa_channel import APPROACHES

logger = logging.getLogger(__name__)

POSITION_MESSAGE_TYPES = {1, 2, 3, 18, 19}
TYPE24_CACHE_TTL = timedelta(minutes=10)


@dataclass(frozen=True)
class DecodedAISMessage:
    """Normalised AIS data, ready for ingestion or further processing."""

    mmsi: str
    message_type: int
    received_at: datetime
    longitude: float | None = None
    latitude: float | None = None
    speed_over_ground: float | None = None
    course_over_ground: float | None = None
    name: str | None = None
    imo_number: str | None = None
    vessel_type: str | None = None
    length_overall: float | None = None
    width: float | None = None
    draft: float | None = None

    @property
    def has_position(self) -> bool:
        return self.longitude is not None and self.latitude is not None


class AISDecoder:
    """Decode NMEA AIS sentences and normalise supported message types."""

    def __init__(self):
        self._type24_parts = {}

    def decode_sentences(
        self,
        sentences: Iterable[str | bytes],
        received_at: datetime | None = None,
    ) -> list[DecodedAISMessage]:
        """Decode a batch of NMEA sentences.

        IterMessages assembles multipart NMEA fragments. AIS type 24 Part A
        and Part B are separate AIS messages and are paired separately here.
        """
        received_at = received_at or datetime.now(UTC)
        encoded_sentences = []

        for sentence in sentences:
            try:
                encoded_sentences.append(
                    sentence
                    if isinstance(sentence, bytes)
                    else sentence.encode("ascii")
                )
            except (UnicodeEncodeError, AttributeError):
                logger.warning("Skipping non-ASCII or invalid AIS sentence.")

        decoded = []

        try:
            for wrapped_message in IterMessages(encoded_sentences):
                try:
                    message = wrapped_message.decode()
                    result = self.decode_message(message, received_at)
                    if result is not None:
                        decoded.append(result)
                except Exception:
                    logger.warning("Skipping an invalid AIS message.", exc_info=True)
        except Exception:
            logger.warning("Unable to process AIS sentence batch.", exc_info=True)

        return decoded

    def decode_message(
        self,
        message,
        received_at: datetime | None = None,
    ) -> DecodedAISMessage | None:
        """Normalise an already-decoded pyais message."""
        received_at = received_at or datetime.now(UTC)

        message_type = self._integer(getattr(message, "msg_type", None))
        mmsi = self._normalise_mmsi(getattr(message, "mmsi", None))

        if message_type is None or mmsi is None:
            return None

        if message_type == 24:
            return self._decode_type24(message, mmsi, received_at)

        if message_type not in POSITION_MESSAGE_TYPES and message_type != 5:
            return None

        longitude = latitude = speed = course = None

        if message_type in POSITION_MESSAGE_TYPES:
            longitude = self._number(getattr(message, "lon", None))
            latitude = self._number(getattr(message, "lat", None))

            if not self._inside_approaches(longitude, latitude):
                longitude = latitude = None

            speed = self._valid_speed(getattr(message, "speed", None))
            course = self._valid_course(getattr(message, "course", None))

        return DecodedAISMessage(
            mmsi=mmsi,
            message_type=message_type,
            received_at=received_at,
            longitude=longitude,
            latitude=latitude,
            speed_over_ground=speed,
            course_over_ground=course,
            **self._static_fields(message),
        )

    def _decode_type24(self, message, mmsi, received_at):
        """Pair type 24 Part A and Part B before emitting static data."""
        self._expire_type24_parts(received_at)

        part_number = self._integer(getattr(message, "partno", None))
        if part_number not in (0, 1):
            return None

        parts = self._type24_parts.setdefault(
            mmsi,
            {"received_at": received_at},
        )
        parts["received_at"] = received_at

        if part_number == 0:
            parts["name"] = self._normalise_name(getattr(message, "shipname", None))
        else:
            static_fields = self._static_fields(message)
            static_fields.pop("name", None)
            parts.update(static_fields)

        if "name" not in parts or "vessel_type" not in parts:
            return None

        result = DecodedAISMessage(
            mmsi=mmsi,
            message_type=24,
            received_at=received_at,
            name=parts.get("name"),
            vessel_type=parts.get("vessel_type"),
            length_overall=parts.get("length_overall"),
            width=parts.get("width"),
        )
        self._type24_parts.pop(mmsi, None)
        return result

    def _expire_type24_parts(self, now):
        expired = [
            mmsi
            for mmsi, parts in self._type24_parts.items()
            if now - parts["received_at"] > TYPE24_CACHE_TTL
        ]
        for mmsi in expired:
            self._type24_parts.pop(mmsi, None)

    @classmethod
    def _static_fields(cls, message):
        ship_type = cls._integer(getattr(message, "ship_type", None))

        to_bow = cls._number(getattr(message, "to_bow", None))
        to_stern = cls._number(getattr(message, "to_stern", None))
        to_port = cls._number(getattr(message, "to_port", None))
        to_starboard = cls._number(getattr(message, "to_starboard", None))

        length = cls._positive_dimension(to_bow, to_stern)
        width = cls._positive_dimension(to_port, to_starboard)

        imo = cls._integer(getattr(message, "imo", None))
        draft = cls._number(getattr(message, "draught", None))

        return {
            "name": cls._normalise_name(getattr(message, "shipname", None)),
            "imo_number": str(imo) if imo and imo > 0 else None,
            "vessel_type": cls._map_vessel_type(ship_type),
            "length_overall": length,
            "width": width,
            "draft": draft if draft is not None and 0 < draft <= 100 else None,
        }

    @staticmethod
    def _normalise_mmsi(value):
        number = AISDecoder._integer(value)
        if number is None or not 100_000_000 <= number <= 999_999_999:
            return None
        return str(number)

    @staticmethod
    def _normalise_name(value):
        if not isinstance(value, str):
            return None
        name = value.replace("@", "").strip()
        return name or None

    @staticmethod
    def _integer(value):
        if isinstance(value, bool):
            return None
        try:
            number = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number

    @staticmethod
    def _number(value):
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if isfinite(number) else None

    @staticmethod
    def _positive_dimension(first, second):
        if first is None or second is None:
            return None
        total = first + second
        return total if 0 < total <= 1000 else None

    @staticmethod
    def _valid_speed(value):
        speed = AISDecoder._number(value)
        # AIS uses 102.3 knots to indicate unavailable speed.
        return speed if speed is not None and 0 <= speed < 102.3 else None

    @staticmethod
    def _valid_course(value):
        course = AISDecoder._number(value)
        # 360 degrees is the AIS unavailable-value sentinel.
        return course if course is not None and 0 <= course < 360 else None

    @staticmethod
    def _inside_approaches(longitude, latitude):
        if longitude is None or latitude is None:
            return False

        bounds = APPROACHES.bounds
        return (
            bounds.west <= longitude <= bounds.east
            and bounds.south <= latitude <= bounds.north
        )

    @staticmethod
    def _map_vessel_type(ship_type):
        if ship_type is None:
            return None
        if 60 <= ship_type <= 69:
            return "passenger"
        if 70 <= ship_type <= 79:
            return "general_cargo"
        if 80 <= ship_type <= 89:
            return "tanker"
        return "other"
