from pathlib import Path
from uuid import uuid4

import copernicusmarine
import xarray as xr

from .base import MarineDataRequest, MarineDataSource


class CopernicusMarineSource(MarineDataSource):
    """Marine data provider backed by Copernicus Marine."""

    DEFAULT_DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"

    def __init__(
        self,
        output_directory: str | Path = "data/raw/copernicus",
        dataset_id: str = DEFAULT_DATASET_ID,
        dataset_version: str | None = None,
    ):
        self.output_directory = Path(output_directory)
        self.dataset_id = dataset_id
        self.dataset_version = dataset_version

    def fetch(self, request: MarineDataRequest) -> xr.Dataset:
        """Download and return a Copernicus Marine dataset subset."""

        self.output_directory.mkdir(parents=True, exist_ok=True)

        filename = f"copernicus_{uuid4().hex}"

        subset_options = {
            "dataset_id": self.dataset_id,
            "variables": list(request.variables),
            "minimum_longitude": request.west,
            "maximum_longitude": request.east,
            "minimum_latitude": request.south,
            "maximum_latitude": request.north,
            "start_datetime": request.start_time,
            "end_datetime": request.end_time,
            "output_directory": str(self.output_directory),
            "output_filename": filename,
            "file_format": "netcdf",
            "coordinates_selection_method": "inside",
        }

        if self.dataset_version is not None:
            subset_options["dataset_version"] = self.dataset_version

        if request.minimum_depth is not None:
            subset_options["minimum_depth"] = request.minimum_depth

        if request.maximum_depth is not None:
            subset_options["maximum_depth"] = request.maximum_depth

        copernicusmarine.subset(**subset_options)

        file_path = self.output_directory / f"{filename}.nc"

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Copernicus download completed but the expected "
                f"NetCDF file was not found: {file_path}"
            )

        with xr.open_dataset(file_path) as dataset:
            return dataset.load()
