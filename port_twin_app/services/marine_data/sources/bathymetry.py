import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import xarray as xr


@dataclass(frozen=True)
class BathymetryRequest:
    """Defines the geographical extent of a bathymetry request."""

    west: float
    south: float
    east: float
    north: float

    def __post_init__(self):
        coordinates = {
            "west": self.west,
            "south": self.south,
            "east": self.east,
            "north": self.north,
        }

        for name, value in coordinates.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite numerical value.")

        if not -180 <= self.west <= 180:
            raise ValueError("Western longitude must be between -180 and 180.")

        if not -180 <= self.east <= 180:
            raise ValueError("Eastern longitude must be between -180 and 180.")

        if not -90 <= self.south <= 90:
            raise ValueError("Southern latitude must be between -90 and 90.")

        if not -90 <= self.north <= 90:
            raise ValueError("Northern latitude must be between -90 and 90.")

        if self.west >= self.east:
            raise ValueError("Western longitude must be less than eastern longitude.")

        if self.south >= self.north:
            raise ValueError("Southern latitude must be less than northern latitude.")


class BathymetryDataSource(ABC):
    """Common interface for static bathymetry data providers."""

    @abstractmethod
    def fetch(self, request: BathymetryRequest) -> xr.Dataset:
        """
        Retrieve bathymetry matching the requested geographical extent.

        Returns:
            An xarray Dataset containing elevation data and coordinates.
        """
        raise NotImplementedError
