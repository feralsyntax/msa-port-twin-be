from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import xarray as xr


@dataclass(frozen=True)
class MarineDataRequest:
    """Defines the geographical and temporal extent of a data request."""

    west: float
    south: float
    east: float
    north: float

    start_time: datetime
    end_time: datetime

    variables: tuple[str, ...]

    minimum_depth: float | None = None
    maximum_depth: float | None = None
    
    def __post_init__(self):
        """Validate geographical, temporal and depth request parameters."""

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

        if self.start_time > self.end_time:
            raise ValueError("Start time must not be later than end time.")

        if not self.variables or any(
            not variable.strip() for variable in self.variables
        ):
            raise ValueError("At least one valid variable must be specified.")

        if self.minimum_depth is not None and self.minimum_depth < 0:
            raise ValueError("Minimum depth cannot be negative.")

        if self.maximum_depth is not None and self.maximum_depth < 0:
            raise ValueError("Maximum depth cannot be negative.")

        if (
            self.minimum_depth is not None
            and self.maximum_depth is not None
            and self.minimum_depth > self.maximum_depth
        ):
            raise ValueError(
                "Minimum depth cannot exceed maximum depth."
            )


class MarineDataSource(ABC):
    """Common interface for external marine data providers."""

    @abstractmethod
    def fetch(self, request: MarineDataRequest) -> xr.Dataset:
        """
        Retrieve marine data matching the supplied request.

        Returns:
            An xarray Dataset containing the requested variables,
            coordinates and associated metadata.
        """
        raise NotImplementedError
