"""Base interface for AIS data sources."""

from abc import ABC, abstractmethod
from collections.abc import Iterator


class AISSourceAdapter(ABC):
    """Common interface implemented by AIS data sources."""

    @abstractmethod
    def iter_batches(self) -> Iterator[list[bytes]]:
        """Yield batches of raw NMEA sentences."""
        raise NotImplementedError
