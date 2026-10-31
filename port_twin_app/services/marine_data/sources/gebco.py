from pathlib import Path

import numpy as np
import xarray as xr

from .bathymetry import BathymetryDataSource, BathymetryRequest


class GEBCOBathymetrySource(BathymetryDataSource):
    """Reads regional bathymetry data from a local GEBCO NetCDF file."""

    def __init__(
        self,
        file_path: str | Path,
        dataset_version: str = "unspecified",
    ):
        self.file_path = Path(file_path)
        self.dataset_version = dataset_version

    def fetch(self, request: BathymetryRequest) -> xr.Dataset:
        """Load and return GEBCO bathymetry for the requested region."""

        if not self.file_path.exists():
            raise FileNotFoundError(f"GEBCO dataset not found: {self.file_path}")

        with xr.open_dataset(self.file_path, engine="scipy") as source:
            longitude_name = self._find_coordinate(source, ("lon", "longitude"))
            latitude_name = self._find_coordinate(source, ("lat", "latitude"))
            elevation_name = self._find_elevation_variable(
                source, longitude_name, latitude_name
            )

            # Extract the elevation variable and normalize coordinate names.
            dataset = source[[elevation_name]].rename(
                {
                    longitude_name: "longitude",
                    latitude_name: "latitude",
                    elevation_name: "elevation_m",
                }
            )

            # Ensure coordinates are ordered before selecting slices.
            dataset = dataset.sortby("longitude").sortby("latitude")

            longitude = dataset.longitude.values
            latitude = dataset.latitude.values

            if longitude.size < 2 or latitude.size < 2:
                raise ValueError(
                    "The GEBCO grid must contain at least two "
                    "coordinates along each spatial dimension."
                )

            # Calculate grid-cell coverage using cell centres
            # and half the coordinate spacing.
            longitude_resolution = float(np.median(np.abs(np.diff(longitude))))
            latitude_resolution = float(np.median(np.abs(np.diff(latitude))))

            # Allow a tiny tolerance for floating-point rounding
            # when comparing requested bounds with grid-cell edges.
            coordinate_tolerance = 1e-10

            grid_west = longitude.min() - longitude_resolution / 2
            grid_east = longitude.max() + longitude_resolution / 2
            grid_south = latitude.min() - latitude_resolution / 2
            grid_north = latitude.max() + latitude_resolution / 2

            if (
                request.west < grid_west - coordinate_tolerance
                or request.east > grid_east + coordinate_tolerance
                or request.south < grid_south - coordinate_tolerance
                or request.north > grid_north + coordinate_tolerance
            ):
                raise ValueError("The requested region extends beyond the GEBCO grid.")

            # Select and load the requested region before closing the file.
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

            elevation = dataset["elevation_m"].values

            if not np.isfinite(elevation).any():
                raise ValueError(
                    "The selected GEBCO region contains no finite elevation values."
                )

            # Preserve the source elevation values and their sign.
            dataset["elevation_m"].attrs.update(
                {
                    "units": source[elevation_name].attrs.get("units", "m"),
                    "source_variable": elevation_name,
                }
            )

            # Preserve source metadata and add ingestion provenance.
            dataset.attrs.update(source.attrs)

            dataset.attrs.update(
                {
                    "provider": "GEBCO",
                    "source_dataset_id": "GEBCO",
                    "source_dataset_version": self.dataset_version,
                    "source_version": source.attrs.get("source", self.dataset_version),
                    "source_file_path": str(self.file_path),
                    "vertical_reference": source.attrs.get(
                        "vertical_datum",
                        source[elevation_name].attrs.get(
                            "standard_name", "unspecified"
                        ),
                    ),
                    "vertical_crs": source.attrs.get(
                        "geospatial_bounds_vertical_crs",
                        "unspecified",
                    ),
                    "elevation_positive": source.attrs.get(
                        "geospatial_vertical_positive",
                        "unspecified",
                    ),
                    "license": source.attrs.get("license", "unspecified"),
                }
            )

        return dataset

    @staticmethod
    def _find_coordinate(
        dataset: xr.Dataset,
        candidates: tuple[str, ...],
    ) -> str:
        """Find a supported one-dimensional spatial coordinate."""

        for name in candidates:
            if name in dataset.coords and dataset[name].ndim == 1:
                return name

        raise ValueError(
            f"GEBCO dataset is missing a supported coordinate: {', '.join(candidates)}."
        )

    @staticmethod
    def _find_elevation_variable(
        dataset: xr.Dataset,
        longitude_name: str,
        latitude_name: str,
    ) -> str:
        """Find the elevation variable using its name and dimensions."""

        candidates = ("elevation", "elev", "Band1")
        expected_dimensions = {longitude_name, latitude_name}

        for name in candidates:
            if name not in dataset.data_vars:
                continue

            if set(dataset[name].dims) == expected_dimensions:
                return name

        raise ValueError(
            "GEBCO dataset does not contain a supported elevation "
            "variable with the expected spatial dimensions."
        )
