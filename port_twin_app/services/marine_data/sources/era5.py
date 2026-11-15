from datetime import UTC
from pathlib import Path

import numpy as np
import xarray as xr

from .base import MarineDataRequest, MarineDataSource


class ERA5WindSource(MarineDataSource):
    """Read and crop a local ERA5 10-metre wind NetCDF file."""

    DEFAULT_DATASET_ID = "reanalysis-era5-single-levels"
    DEFAULT_DATASET_VERSION = "ERA5"
    DEFAULT_SOURCE_VERSION = "ERA5 hourly reanalysis"

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)

    def fetch(self, request: MarineDataRequest) -> xr.Dataset:
        if not self.file_path.is_file():
            raise FileNotFoundError(f"ERA5 NetCDF file not found: {self.file_path}")

        if set(request.variables) != {"u10", "v10"} or len(request.variables) != 2:
            raise ValueError(
                "ERA5 wind requests must specify variables=('u10', 'v10')."
            )

        with xr.open_dataset(self.file_path) as source:
            dataset = source

            # CDS downloads may use valid_time, while other files use time.
            rename = {}
            for source_name, target_name in (
                ("valid_time", "time"),
                ("lat", "latitude"),
                ("lon", "longitude"),
            ):
                if source_name in dataset.coords and target_name not in dataset.coords:
                    rename[source_name] = target_name

            if rename:
                dataset = dataset.rename(rename)

            required_coordinates = {"time", "latitude", "longitude"}
            if not required_coordinates.issubset(dataset.coords):
                raise ValueError(
                    "ERA5 file must contain time, latitude and longitude coordinates."
                )

            missing_variables = {"u10", "v10"} - set(dataset.data_vars)
            if missing_variables:
                raise ValueError(
                    f"ERA5 file is missing wind variables: {sorted(missing_variables)}"
                )

            start_time = request.start_time
            end_time = request.end_time

            if start_time.tzinfo is not None:
                start_time = start_time.astimezone(UTC).replace(tzinfo=None)

            if end_time.tzinfo is not None:
                end_time = end_time.astimezone(UTC).replace(tzinfo=None)

            dataset = dataset[["u10", "v10"]].where(
                (dataset.longitude >= request.west)
                & (dataset.longitude <= request.east)
                & (dataset.latitude >= request.south)
                & (dataset.latitude <= request.north),
                drop=True,
            )

            dataset = (
                dataset.where(
                    (dataset.time >= np.datetime64(start_time))
                    & (dataset.time <= np.datetime64(end_time)),
                    drop=True,
                )
                .sortby("latitude")
                .sortby("longitude")
                .sortby("time")
                .load()
            )

            if any(
                dataset.sizes.get(axis, 0) == 0
                for axis in ("time", "latitude", "longitude")
            ):
                raise ValueError(
                    "The requested extent contains no ERA5 wind observations."
                )

        dataset.attrs.update(
            {
                "provider": "Copernicus Climate Data Store",
                "source_dataset_id": self.DEFAULT_DATASET_ID,
                "source_dataset_version": self.DEFAULT_DATASET_VERSION,
                "source_version": self.DEFAULT_SOURCE_VERSION,
                "source_file_path": str(self.file_path),
            }
        )

        return dataset
