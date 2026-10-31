import numpy as np
import xarray as xr


def normalize_copernicus_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """
    Normalize Copernicus ocean data into the marine AI conventions.

    Output variables:
        sea_surface_height_m: Sea surface height above geoid in metres.
        current_u_ms: Eastward current velocity in m/s.
        current_v_ms: Northward current velocity in m/s.

    Missing values are preserved.
    """

    required_variables = {"zos", "uo", "vo"}
    missing_variables = required_variables - set(dataset.data_vars)

    expected_units = {
        "zos": "m",
        "uo": "m s-1",
        "vo": "m s-1",
    }

    if missing_variables:
        raise ValueError(
            f"Dataset is missing required variables: {sorted(missing_variables)}"
        )

    # Validate source units before normalization.
    for variable, expected_unit in expected_units.items():
        actual_unit = dataset[variable].attrs.get("units")

        if actual_unit != expected_unit:
            raise ValueError(
                f"{variable} has unsupported units: {actual_unit}. "
                f"Expected {expected_unit}."
            )

    normalized = dataset[["zos", "uo", "vo"]].copy()

    # Remove the vertical dimension from near-surface current fields.
    for variable in ("uo", "vo"):
        if "depth" in normalized[variable].dims:
            if normalized.sizes["depth"] != 1:
                raise ValueError(
                    f"{variable} contains multiple depth levels. "
                    "Select a depth before normalization."
                )

            normalized[variable] = normalized[variable].isel(
                depth=0,
                drop=True,
            )

    # Remove the standalone depth coordinate after extracting
    # the selected near-surface level.
    if "depth" in normalized.coords:
        normalized = normalized.drop_vars("depth")

    normalized = normalized.rename(
        {
            "zos": "sea_surface_height_m",
            "uo": "current_u_ms",
            "vo": "current_v_ms",
        }
    )

    expected_dimensions = ("time", "latitude", "longitude")

    for variable in (
        "sea_surface_height_m",
        "current_u_ms",
        "current_v_ms",
    ):
        if normalized[variable].dims != expected_dimensions:
            raise ValueError(
                f"{variable} has unexpected dimensions: {normalized[variable].dims}"
            )

        values = normalized[variable].values
        valid_values = values[np.isfinite(values)]

        if valid_values.size == 0:
            raise ValueError(f"{variable} contains no valid numerical values.")

    # Ensure all variables share the same valid-data mask.
    masks = [
        np.isfinite(normalized[variable].values)
        for variable in (
            "sea_surface_height_m",
            "current_u_ms",
            "current_v_ms",
        )
    ]

    if not all(np.array_equal(masks[0], mask) for mask in masks[1:]):
        raise ValueError("Marine variables have inconsistent missing-data masks.")

    normalized["sea_surface_height_m"].attrs["units"] = "m"
    normalized["current_u_ms"].attrs["units"] = "m s-1"
    normalized["current_v_ms"].attrs["units"] = "m s-1"

    normalized.attrs["normalization_source"] = "Copernicus Marine"
    normalized.attrs["vertical_reference"] = "geoid"

    return normalized
