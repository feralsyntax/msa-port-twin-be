from datetime import UTC
from pathlib import Path

import numpy as np
import xarray as xr

from .base import MarineDataRequest, MarineDataSource


class GriddedAltimetrySource(MarineDataSource):
    """Read and crop a local gridded DUACS L4 altimetry NetCDF file."""

    DEFAULT_DATASET_ID = "cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D"

    def __init__(
        self,
        file_path: str | Path,
        dataset_version: str = "202506",
    ):
        self.file_path = Path(file_path)
        self.dataset_version = dataset_version

    def fetch(self, request: MarineDataRequest) -> xr.Dataset:
        if not self.file_path.is_file():
            raise FileNotFoundError(
                f"Altimetry NetCDF file not found: {self.file_path}"
            )

        with xr.open_dataset(self.file_path) as source:
            if "sla" not in source.data_vars:
                raise ValueError("Altimetry file is missing the sla variable.")

            title = source.attrs.get("title", "")
            processing_level = source.attrs.get("processing_level", "")

            if "DUACS" not in title.upper() or processing_level != "L4":
                raise ValueError("Expected a DUACS Level-4 gridded altimetry product.")

            if not {"time", "latitude", "longitude"}.issubset(source.coords):
                raise ValueError(
                    "Altimetry file must contain time, latitude and longitude."
                )

            if request.variables != ("sla",):
                raise ValueError(
                    "Gridded altimetry requests must specify variables=('sla',)."
                )

            start_time = request.start_time
            end_time = request.end_time

            if start_time.tzinfo is not None:
                start_time = start_time.astimezone(UTC).replace(tzinfo=None)

            if end_time.tzinfo is not None:
                end_time = end_time.astimezone(UTC).replace(tzinfo=None)

            dataset = source[["sla"]].where(
                (source.longitude >= request.west)
                & (source.longitude <= request.east)
                & (source.latitude >= request.south)
                & (source.latitude <= request.north),
                drop=True,
            )

            dataset = dataset.where(
                (dataset.time >= np.datetime64(start_time))
                & (dataset.time <= np.datetime64(end_time)),
                drop=True,
            ).load()

            if (
                not dataset.sizes.get("time")
                or not dataset.sizes.get("latitude")
                or not dataset.sizes.get("longitude")
            ):
                raise ValueError(
                    "The requested extent contains no altimetry observations."
                )

        dataset.attrs.update(
            {
                "provider": "Copernicus Marine",
                "source_dataset_id": self.DEFAULT_DATASET_ID,
                "source_dataset_version": self.dataset_version,
                "source_version": dataset.attrs.get("product_version", "unspecified"),
                "source_file_path": str(self.file_path),
            }
        )

        return dataset
