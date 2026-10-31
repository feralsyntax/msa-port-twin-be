from pathlib import Path

import numpy as np
import xarray as xr

from .bathymetry import BathymetryDataSource, BathymetryRequest


class GEBCOBathymetrySource(BathymetryDataSource):
    """Bathymetry provider backed by a local GEBCO NetCDF file."""

    def __init__(
        self,
        file_path: str | Path,
        dataset_version: str = "unspecified",
    ):
        self.file_path = Path(file_path)
        self.dataset_version = dataset_version

    def fetch(self, request: BathymetryRequest) -> xr.Dataset:
        """Read and return a geographical subset of the GEBCO grid."""

        if not self.file_path.is_file():
            raise FileNotFoundError(f"GEBCO NetCDF file not found: {self.file_path}")

        with xr.open_dataset(self.file_path) as source:
            longitude_name = self._find_coordinate(source, ("lon", "longitude"))
            latitude_name = self._find_coordinate(source, ("lat", "latitude"))

            elevation_name = self._find_elevation_variable(
                source,
                longitude_name,
                latitude_name,
            )

            dataset = source[[elevation_name]].copy()

            # Normalize coordinate names.
            rename = {}

            if longitude_name != "longitude":
                rename[longitude_name] = "longitude"

            if latitude_name != "latitude":
                rename[latitude_name] = "latitude"

            if elevation_name != "elevation_m":
                rename[elevation_name] = "elevation_m"

            dataset = dataset.rename(rename)

            # Ensure coordinates are ordered before selecting slices.
            dataset = dataset.sortby("longitude").sortby("latitude")

            longitude = dataset.longitude.values
            latitude = dataset.latitude.values

            if (
                request.west < longitude.min()
                or request.east > longitude.max()
                or request.south < latitude.min()
                or request.north > latitude.max()
            ):
                raise ValueError("The requested region extends beyond the GEBCO grid.")

            dataset = dataset.sel(
                longitude=slice(request.west, request.east),
                latitude=slice(request.south, request.north),
            ).load()

            if dataset.sizes["longitude"] < 2:
                raise ValueError(
                    "The selected region contains fewer than two longitude coordinates."
                )

            if dataset.sizes["latitude"] < 2:
                raise ValueError(
                    "The selected region contains fewer than two latitude coordinates."
                )

            values = dataset["elevation_m"].values

            if not np.isfinite(values).any():
                raise ValueError(
                    "The selected GEBCO region contains no valid elevations."
                )

            dataset["elevation_m"].attrs.update(
                {
                    "units": source[elevation_name].attrs.get("units", "m"),
                    "source_variable": elevation_name,
                }
            )

            dataset.attrs.update(source.attrs)
            dataset.attrs.update(
                {
                    "provider": "GEBCO",
                    "source_dataset_id": "GEBCO",
                    "source_dataset_version": self.dataset_version,
                    "source_version": source.attrs.get("source", self.dataset_version),
                    "source_file_path": str(self.file_path),
                    "vertical_reference": source.attrs.get(
                        "vertical_datum", "unspecified"
                    ),
                }
            )

            return dataset

    @staticmethod
    def _find_coordinate(
        dataset: xr.Dataset,
        candidates: tuple[str, ...],
    ) -> str:
        for name in candidates:
            if name in dataset.coords:
                if dataset[name].ndim != 1:
                    raise ValueError(f"Coordinate {name} must be one-dimensional.")

                return name

        raise ValueError(f"Could not find a coordinate matching {candidates}.")

    @staticmethod
    def _find_elevation_variable(
        dataset: xr.Dataset,
        longitude_name: str,
        latitude_name: str,
    ) -> str:
        candidates = ("elevation", "elev", "Band1")

        for name in candidates:
            if name in dataset.data_vars:
                variable = dataset[name]

                if set(variable.dims) == {
                    longitude_name,
                    latitude_name,
                }:
                    return name

        raise ValueError("Could not identify a supported GEBCO elevation variable.")
