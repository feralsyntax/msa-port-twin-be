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
