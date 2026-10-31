import numpy as np
import xarray as xr

REQUIRED_VARIABLES = {
    "water_level_m",
    "current_u_ms",
    "current_v_ms",
}

REQUIRED_COORDINATES = {
    "time",
    "latitude",
    "longitude",
}


def validate_marine_dataset(dataset: xr.Dataset) -> None:
    """Validate a normalized marine dataset."""

    missing_variables = REQUIRED_VARIABLES - set(dataset.data_vars)
    missing_coordinates = REQUIRED_COORDINATES - set(dataset.coords)

    if missing_variables:
        raise ValueError(f"Missing required variables: {sorted(missing_variables)}")

    if missing_coordinates:
        raise ValueError(f"Missing required coordinates: {sorted(missing_coordinates)}")

    if dataset.sizes["time"] == 0:
        raise ValueError("Dataset contains no time steps.")

    if dataset.sizes["latitude"] < 2 or dataset.sizes["longitude"] < 2:
        raise ValueError("Spatial grid must contain at least 2 points per axis.")

    times = dataset.time.values
    if not np.issubdtype(times.dtype, np.datetime64):
        raise ValueError("Time coordinates must use datetime64 values.")

    if np.isnat(times).any():
        raise ValueError("Time coordinates contain invalid timestamps.")

    if np.any(np.diff(times) <= np.timedelta64(0, "ns")):
        raise ValueError("Time coordinates must be strictly increasing.")

    for coordinate in ("latitude", "longitude"):
        values = dataset[coordinate].values

        if not np.isfinite(values).all():
            raise ValueError(f"{coordinate} contains non-finite values.")

        if np.any(np.diff(values) <= 0):
            raise ValueError(f"{coordinate} must be strictly increasing.")

    latitude = dataset.latitude.values
    longitude = dataset.longitude.values

    if np.any((latitude < -90) | (latitude > 90)):
        raise ValueError("Latitude values must be between -90 and 90.")

    if np.any((longitude < -180) | (longitude > 360)):
        raise ValueError("Longitude values fall outside supported bounds.")

    reference_mask = None

    for variable in REQUIRED_VARIABLES:
        data = dataset[variable]

        if data.dims != ("time", "latitude", "longitude"):
            raise ValueError(f"{variable} has unexpected dimensions: {data.dims}")

        if data.shape != (
            dataset.sizes["time"],
            dataset.sizes["latitude"],
            dataset.sizes["longitude"],
        ):
            raise ValueError(f"{variable} has an invalid shape.")

        values = data.values

        if np.isinf(values).any():
            raise ValueError(f"{variable} contains infinite values.")

        valid_mask = np.isfinite(values)

        if not valid_mask.any():
            raise ValueError(f"{variable} contains no valid observations.")

        if reference_mask is None:
            reference_mask = valid_mask
        elif not np.array_equal(reference_mask, valid_mask):
            raise ValueError("Marine variables have inconsistent missing-data masks.")
