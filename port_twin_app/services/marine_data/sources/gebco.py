from pathlib import Path

import numpy as np
import xarray as xr

from .bathymetry import BathymetryDataSource, BathymetryRequest


class GEBCOBathymetrySource(BathymetryDataSource):
    """Reads regional bathymetry and optional TID data from GEBCO NetCDF files."""

    def __init__(
        self,
        file_path: str | Path,
        dataset_version: str = "unspecified",
        tid_file_path: str | Path | None = None,
    ):
        self.file_path = Path(file_path)
        self.dataset_version = dataset_version
        self.tid_file_path = Path(tid_file_path) if tid_file_path else None

    def fetch(self, request: BathymetryRequest) -> xr.Dataset:
        """Load and return GEBCO bathymetry and optional TID for a region."""

        if not self.file_path.exists():
            raise FileNotFoundError(f"GEBCO dataset not found: {self.file_path}")

        if self.tid_file_path and not self.tid_file_path.exists():
            raise FileNotFoundError(
                f"GEBCO TID dataset not found: {self.tid_file_path}"
            )

        with xr.open_dataset(self.file_path, engine="scipy") as source:
            longitude_name = self._find_coordinate(source, ("lon", "longitude"))
            latitude_name = self._find_coordinate(source, ("lat", "latitude"))
            elevation_name = self._find_elevation_variable(
                source, longitude_name, latitude_name
            )

            dataset = source[[elevation_name]].rename(
                {
                    longitude_name: "longitude",
                    latitude_name: "latitude",
                    elevation_name: "elevation_m",
                }
            )

            dataset = dataset.sortby("longitude").sortby("latitude")

            longitude = dataset.longitude.values
            latitude = dataset.latitude.values

            if longitude.size < 2 or latitude.size < 2:
                raise ValueError(
                    "The GEBCO grid must contain at least two "
                    "coordinates along each spatial dimension."
                )

            self._validate_grid_coverage(longitude, latitude, request)

            dataset = dataset.sel(
                longitude=slice(request.west, request.east),
                latitude=slice(request.south, request.north),
            ).load()

            self._validate_selected_grid(dataset)

            elevation = dataset["elevation_m"].values

            if not np.isfinite(elevation).any():
                raise ValueError(
                    "The selected GEBCO region contains no finite elevation values."
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

        if self.tid_file_path:
            tid_dataset = self._fetch_tid(
                request=request,
                expected_longitude=dataset.longitude.values,
                expected_latitude=dataset.latitude.values,
            )

            dataset["tid"] = tid_dataset["tid"]
            dataset["tid"].attrs.update(tid_dataset["tid"].attrs)
            dataset.attrs["tid_source_file_path"] = str(self.tid_file_path)

        return dataset

    def _fetch_tid(
        self,
        request: BathymetryRequest,
        expected_longitude: np.ndarray,
        expected_latitude: np.ndarray,
    ) -> xr.Dataset:
        """Load TID classifications and verify alignment with bathymetry."""

        with xr.open_dataset(self.tid_file_path, engine="scipy") as source:
            longitude_name = self._find_coordinate(source, ("lon", "longitude"))
            latitude_name = self._find_coordinate(source, ("lat", "latitude"))
            tid_name = self._find_tid_variable(source, longitude_name, latitude_name)

            dataset = source[[tid_name]].rename(
                {
                    longitude_name: "longitude",
                    latitude_name: "latitude",
                    tid_name: "tid",
                }
            )

            dataset = dataset.sortby("longitude").sortby("latitude")

            if not np.array_equal(dataset.longitude.values, self._source_longitude):
                raise ValueError(
                    "GEBCO TID longitude coordinates do not match bathymetry."
                )

            if not np.array_equal(dataset.latitude.values, self._source_latitude):
                raise ValueError(
                    "GEBCO TID latitude coordinates do not match bathymetry."
                )

            dataset = dataset.sel(
                longitude=slice(request.west, request.east),
                latitude=slice(request.south, request.north),
            ).load()

            if not np.array_equal(dataset.longitude.values, expected_longitude):
                raise ValueError(
                    "Selected GEBCO TID longitudes do not match bathymetry."
                )

            if not np.array_equal(dataset.latitude.values, expected_latitude):
                raise ValueError(
                    "Selected GEBCO TID latitudes do not match bathymetry."
                )

            tid_values = dataset["tid"].values

            if not np.isfinite(tid_values).all():
                raise ValueError("GEBCO TID data contains non-finite values.")

            if not np.equal(tid_values, np.floor(tid_values)).all():
                raise ValueError("GEBCO TID classifications must be integers.")

            dataset["tid"] = dataset["tid"].astype(np.int16)
            dataset["tid"].attrs.update(source[tid_name].attrs)
            dataset["tid"].attrs["source_variable"] = tid_name

        return dataset

    @staticmethod
    def _validate_grid_coverage(
        longitude: np.ndarray,
        latitude: np.ndarray,
        request: BathymetryRequest,
    ) -> None:
        """Ensure the requested region is covered by source grid cells."""

        longitude_resolution = float(np.median(np.abs(np.diff(longitude))))
        latitude_resolution = float(np.median(np.abs(np.diff(latitude))))

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

    @staticmethod
    def _validate_selected_grid(dataset: xr.Dataset) -> None:
        if dataset.sizes["longitude"] < 2:
            raise ValueError(
                "The selected region contains fewer than two longitude coordinates."
            )

        if dataset.sizes["latitude"] < 2:
            raise ValueError(
                "The selected region contains fewer than two latitude coordinates."
            )

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

    @staticmethod
    def _find_tid_variable(
        dataset: xr.Dataset,
        longitude_name: str,
        latitude_name: str,
    ) -> str:
        """Find the TID variable using its name and spatial dimensions."""

        expected_dimensions = {longitude_name, latitude_name}

        if (
            "tid" in dataset.data_vars
            and set(dataset["tid"].dims) == expected_dimensions
        ):
            return "tid"

        raise ValueError(
            "GEBCO TID dataset does not contain a TID variable "
            "with the expected spatial dimensions."
        )
