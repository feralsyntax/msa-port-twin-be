"""Prepare projected computational grids from source GEBCO bathymetry."""

import json
from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import CRS, Transformer
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from port_twin_app.services.marine_data.domains.mombasa_channel import (
    GEOGRAPHIC_CRS,
    PROJECTED_CRS,
    get_zone,
)
from port_twin_app.services.marine_data.preparation.mombasa_bathymetry import (
    DEFAULT_OUTPUT_DIR,
)

PROJECTED_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "projected"

SOURCE_FILES = {
    "approaches": DEFAULT_OUTPUT_DIR / "approaches_gebco_2026.nc",
    "port_operations": DEFAULT_OUTPUT_DIR / "port_operations_gebco_2026.nc",
}


def _aligned_axis(minimum, maximum, resolution):
    """Create grid coordinates aligned to a shared zero-metre origin."""
    start = np.ceil(minimum / resolution) * resolution
    stop = np.floor(maximum / resolution) * resolution

    if stop < start:
        raise ValueError("Projected source extent is smaller than grid spacing.")

    values = np.arange(start, stop + resolution * 0.5, resolution)

    if values.size < 2:
        raise ValueError("Projected grid must contain at least two points per axis.")

    return values.astype(np.float64)


def _source_resolution(dataset):
    """Return source geographic spacing in arc-seconds."""
    return {
        coordinate: float(np.median(np.abs(np.diff(dataset[coordinate].values))) * 3600)
        for coordinate in ("longitude", "latitude")
    }


def _load_source(zone_name, source_dataset=None):
    if source_dataset is not None:
        return source_dataset.copy(deep=True)

    try:
        source_path = SOURCE_FILES[zone_name]
    except KeyError:
        raise ValueError(f"No source file configured for zone: {zone_name}") from None

    if not source_path.is_file():
        raise FileNotFoundError(f"Prepared GEBCO source not found: {source_path}")

    with xr.open_dataset(source_path) as dataset:
        return dataset.load()


def build_projected_bathymetry_grid(zone_name, source_dataset=None):
    """Build a projected bathymetry grid without modifying source data."""
    zone = get_zone(zone_name)
    source = _load_source(zone_name, source_dataset)

    required = {"elevation_m", "bathymetry_m", "tid"}
    missing = required - set(source.data_vars)

    if missing:
        raise ValueError(f"Source dataset is missing variables: {sorted(missing)}")

    if source.elevation_m.dims != ("latitude", "longitude"):
        raise ValueError("Source elevation must use latitude/longitude dimensions.")

    if source.tid.dims != ("latitude", "longitude"):
        raise ValueError("Source TID must use latitude/longitude dimensions.")

    if source.elevation_m.shape != source.tid.shape:
        raise ValueError("Source elevation and TID grids must have matching shapes.")

    longitude = source.longitude.values
    latitude = source.latitude.values
    elevation = source.elevation_m.values.astype(np.float64)
    tid = source.tid.values

    if (
        longitude.size < 2
        or latitude.size < 2
        or not np.isfinite(longitude).all()
        or not np.isfinite(latitude).all()
        or np.any(np.diff(longitude) <= 0)
        or np.any(np.diff(latitude) <= 0)
    ):
        raise ValueError("Source geographic coordinates are invalid.")

    if np.isinf(elevation).any():
        raise ValueError("Source elevation contains infinite values.")

    if not np.isfinite(elevation).any():
        raise ValueError("Source elevation contains no valid values.")

    if np.isinf(tid.astype(np.float64)).any():
        raise ValueError("Source TID contains infinite values.")

    source_xx, source_yy = np.meshgrid(longitude, latitude)

    transformer = Transformer.from_crs(
        GEOGRAPHIC_CRS,
        PROJECTED_CRS,
        always_xy=True,
    )

    source_easting, source_northing = transformer.transform(
        source_xx,
        source_yy,
    )

    source_points = np.column_stack((source_easting.ravel(), source_northing.ravel()))

    # Only underwater source cells may contribute to bathymetry.
    valid_water = np.isfinite(source["bathymetry_m"].values)

    valid_tid = valid_water & np.isfinite(tid.astype(np.float64))

    if valid_water.sum() < 3:
        raise ValueError("Insufficient underwater source points.")

    if not valid_tid.any():
        raise ValueError("No valid underwater TID classifications.")

    resolution = zone.target_resolution_m

    easting = _aligned_axis(
        np.nanmin(source_easting),
        np.nanmax(source_easting),
        resolution,
    )
    northing = _aligned_axis(
        np.nanmin(source_northing),
        np.nanmax(source_northing),
        resolution,
    )

    target_xx, target_yy = np.meshgrid(easting, northing)
    target_points = (target_xx, target_yy)

    # Interpolate only underwater bathymetry.
    water_points = source_points[valid_water.ravel()]
    water_depths = source["bathymetry_m"].values.ravel()[valid_water.ravel()]

    interpolated_bathymetry = griddata(
        water_points,
        water_depths,
        target_points,
        method="linear",
    )

    # Classify each target cell using its nearest original source cell.
    # This prevents interpolation across source cells classified as land.
    source_tree = cKDTree(source_points)
    _, nearest_source_indices = source_tree.query(
        np.column_stack((target_xx.ravel(), target_yy.ravel()))
    )

    nearest_source_is_water = valid_water.ravel()[nearest_source_indices]
    nearest_source_is_water = nearest_source_is_water.reshape(target_xx.shape)

    source_coverage = nearest_source_is_water & np.isfinite(interpolated_bathymetry)

    # Never retain interpolated values outside supported underwater coverage.
    interpolated_bathymetry = np.where(
        source_coverage,
        interpolated_bathymetry,
        np.nan,
    )

    interpolated_elevation = -interpolated_bathymetry

    # Preserve original categorical TID values using nearest water source cells.
    interpolated_tid = griddata(
        water_points,
        tid.ravel()[valid_tid.ravel()].astype(np.float64),
        target_points,
        method="nearest",
    )

    interpolated_tid = np.where(
        source_coverage,
        interpolated_tid,
        np.nan,
    )

    bathymetry = interpolated_bathymetry

    crs = CRS.from_epsg(32737)
    crs_attributes = {
        **crs.to_cf(),
        "spatial_ref": crs.to_wkt(),
        "crs_wkt": crs.to_wkt(),
    }

    grid = xr.Dataset(
        data_vars={
            "elevation_m": (
                ("northing", "easting"),
                interpolated_elevation.astype(np.float32),
                {
                    "long_name": "Interpolated GEBCO elevation",
                    "units": "m",
                    "positive": "up",
                    "source_variable": "elevation_m",
                    "interpolation_method": "linear",
                    "grid_mapping": "spatial_ref",
                },
            ),
            "bathymetry_m": (
                ("northing", "easting"),
                bathymetry.astype(np.float32),
                {
                    "long_name": "Interpolated seafloor depth",
                    "units": "m",
                    "positive": "down",
                    "source_variable": "elevation_m",
                    "interpolation_method": "linear",
                    "grid_mapping": "spatial_ref",
                },
            ),
            "tid": (
                ("northing", "easting"),
                interpolated_tid.astype(np.float32),
                {
                    **source.tid.attrs,
                    "long_name": "GEBCO Type Identifier",
                    "resampling_method": "nearest",
                    "grid_mapping": "spatial_ref",
                },
            ),
            "source_coverage": (
                ("northing", "easting"),
                source_coverage,
                {
                    "long_name": "Within source interpolation coverage",
                    "description": (
                        "True only where the nearest source grid cell is underwater "
                        "and linear bathymetry interpolation is supported."
                    ),
                    "grid_mapping": "spatial_ref",
                },
            ),
            "spatial_ref": ((), np.int32(0), crs_attributes),
        },
        coords={
            "easting": (
                "easting",
                easting,
                {
                    "standard_name": "projection_x_coordinate",
                    "units": "m",
                },
            ),
            "northing": (
                "northing",
                northing,
                {
                    "standard_name": "projection_y_coordinate",
                    "units": "m",
                },
            ),
        },
        attrs={
            "provider": source.attrs.get("provider", "GEBCO"),
            "dataset_id": source.attrs.get("source_dataset_id", "GEBCO"),
            "dataset_version": source.attrs.get(
                "source_dataset_version",
                "GEBCO_2026",
            ),
            "domain_name": zone.name,
            "domain_source_reference": zone.source_reference,
            "geographic_crs": GEOGRAPHIC_CRS,
            "projected_crs": PROJECTED_CRS,
            "grid_resolution_m": resolution,
            "grid_alignment_origin_m": 0,
            "grid_dimensions": "northing, easting",
            "source_resolution_longitude_arc_seconds": (
                _source_resolution(source)["longitude"]
            ),
            "source_resolution_latitude_arc_seconds": (
                _source_resolution(source)["latitude"]
            ),
            "source_vertical_reference": source.attrs.get(
                "source_vertical_reference",
                source.attrs.get("vertical_reference", "unspecified"),
            ),
            "vertical_datum_correction_applied": False,
            "vertical_datum_correction_m": 0.0,
            "bathymetry_positive_direction": "down",
            "elevation_interpolation": "linear",
            "tid_resampling": "nearest",
            "source_accuracy_claim": False,
            "source_accuracy_note": (
                "Target grid spacing does not represent source-data accuracy."
            ),
        },
    )

    if not source_coverage.any():
        raise ValueError("Projected grid contains no source interpolation coverage.")

    if not np.isfinite(bathymetry).any():
        raise ValueError("Projected grid contains no valid underwater bathymetry.")

    return grid


def prepare_projected_bathymetry_grid(zone_name, output_dir=None):
    """Build and save a projected grid with its provenance metadata."""
    zone = get_zone(zone_name)
    output_dir = Path(output_dir or PROJECTED_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = build_projected_bathymetry_grid(zone_name)

    output_path = output_dir / (
        f"{zone.name}_gebco_2026_utm37s_{zone.target_resolution_m}m.nc"
    )
    provenance_path = output_path.with_suffix(".provenance.json")

    grid.to_netcdf(output_path, engine="scipy")

    provenance = {
        "provider": grid.attrs["provider"],
        "dataset_id": grid.attrs["dataset_id"],
        "dataset_version": grid.attrs["dataset_version"],
        "domain_name": zone.name,
        "source_file": str(SOURCE_FILES[zone.name]),
        "output_file": str(output_path),
        "geographic_crs": GEOGRAPHIC_CRS,
        "projected_crs": PROJECTED_CRS,
        "grid_resolution_m": zone.target_resolution_m,
        "grid_alignment_origin_m": 0,
        "grid_width": grid.sizes["easting"],
        "grid_height": grid.sizes["northing"],
        "source_resolution": {
            "longitude_arc_seconds": grid.attrs[
                "source_resolution_longitude_arc_seconds"
            ],
            "latitude_arc_seconds": grid.attrs[
                "source_resolution_latitude_arc_seconds"
            ],
        },
        "interpolation": {
            "elevation": "linear",
            "tid": "nearest",
            "extrapolation": False,
        },
        "source_coverage_cells": int(grid.source_coverage.sum().item()),
        "valid_bathymetry_cells": int(np.isfinite(grid.bathymetry_m.values).sum()),
        "vertical_reference": grid.attrs["source_vertical_reference"],
        "vertical_datum_correction_applied": False,
        "vertical_datum_correction_m": None,
        "source_accuracy_claim": False,
        "output_variables": list(grid.data_vars),
    }

    provenance_path.write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )

    return {
        "dataset": grid,
        "output_path": output_path,
        "provenance_path": provenance_path,
    }
