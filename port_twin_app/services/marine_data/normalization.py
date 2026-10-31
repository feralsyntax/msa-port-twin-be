import numpy as np
import xarray as xr


def normalize_copernicus_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """
    Normalize Copernicus ocean data into the marine AI conventions.

    Output variables:
        water_level_m: Sea surface height in metres.
        current_u_ms: Eastward current velocity in m/s.
        current_v_ms: Northward current velocity in m/s.

    Missing values are preserved.
    """

    required_variables = {"zos", "uo", "vo"}
    missing_variables = required_variables - set(dataset.data_vars)

    if missing_variables:
        raise ValueError(
            f"Dataset is missing required variables: {sorted(missing_variables)}"
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

    normalized = normalized.rename(
        {
            "zos": "water_level_m",
            "uo": "current_u_ms",
            "vo": "current_v_ms",
        }
    )

    expected_dimensions = ("time", "latitude", "longitude")

    for variable in (
        "water_level_m",
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
            "water_level_m",
            "current_u_ms",
            "current_v_ms",
        )
    ]

    if not all(np.array_equal(masks[0], mask) for mask in masks[1:]):
        raise ValueError("Marine variables have inconsistent missing-data masks.")

    normalized["water_level_m"].attrs["units"] = "m"
    normalized["current_u_ms"].attrs["units"] = "m s-1"
    normalized["current_v_ms"].attrs["units"] = "m s-1"

    normalized.attrs["normalization_source"] = "Copernicus Marine"
    normalized.attrs["vertical_reference"] = "geoid"

    return normalized
