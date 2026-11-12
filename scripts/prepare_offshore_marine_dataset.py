"""Prepare an offshore regional marine dataset for FNO training."""

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from port_twin_app.services.marine_ai.preparation import (
    prepare_marine_dataset,
)
from port_twin_app.services.marine_ai.serialization import (
    load_prepared_dataset,
    save_prepared_dataset,
)
from port_twin_app.services.marine_data.normalization import (
    normalize_copernicus_dataset,
    normalize_gebco_bathymetry,
)

DATA_ROOT = PROJECT_ROOT / "data"

COPERNICUS_FILE = (
    DATA_ROOT / "raw/copernicus/copernicus_c3aba26804f0488e86a8513e71ac2a57.nc"
)
COPERNICUS_PROVENANCE = COPERNICUS_FILE.with_suffix(".provenance.json")

GEBCO_FILE = DATA_ROOT / "raw/gebco/gebco_2026_mombasa_extended.nc"
GEBCO_TID_FILE = DATA_ROOT / "raw/gebco/gebco_2026_mombasa_extended_tid.nc"
GEBCO_PROVENANCE = GEBCO_FILE.with_suffix(".provenance.json")

OUTPUT_FILE = DATA_ROOT / "processed/marine/offshore_training_dataset.npz"

# Offshore regional domain east of Mombasa.
WEST = 40.16667
EAST = 41.00001
SOUTH = -5.00001
NORTH = -3.08000


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def prepare_copernicus_grid() -> tuple[xr.Dataset, np.ndarray, np.ndarray]:
    """Load and normalize Copernicus data for the selected regional grid."""

    provenance = load_json(COPERNICUS_PROVENANCE)

    with xr.open_dataset(COPERNICUS_FILE) as source:
        source = source.load()

    latitude = source.latitude.values
    longitude = source.longitude.values

    target_latitude = latitude[(latitude >= SOUTH) & (latitude <= NORTH)]
    target_longitude = longitude[(longitude >= WEST) & (longitude <= EAST)]

    if len(target_latitude) == 0 or len(target_longitude) == 0:
        raise ValueError("The selected domain contains no Copernicus grid cells.")

    marine = source.sel(
        latitude=target_latitude,
        longitude=target_longitude,
    ).copy()

    marine.attrs.update(
        {
            "provider": provenance["provider"],
            "source_dataset_id": provenance["dataset_id"],
            "source_dataset_version": provenance["dataset_version"],
            "source_version": provenance["source_version"],
            "license_name": provenance["license_name"],
            "source_file_path": str(COPERNICUS_FILE.relative_to(PROJECT_ROOT)),
        }
    )

    marine = normalize_copernicus_dataset(marine)

    return marine, target_latitude, target_longitude


def prepare_gebco_grid(
    target_latitude: np.ndarray,
    target_longitude: np.ndarray,
) -> xr.Dataset:
    """Interpolate GEBCO elevation and TID onto the Copernicus grid."""

    provenance = load_json(GEBCO_PROVENANCE)

    with xr.open_dataset(GEBCO_FILE) as source:
        elevation_source = source.load()

    with xr.open_dataset(GEBCO_TID_FILE) as source:
        tid_source = source.load()

    # Standardize coordinate names and ordering.
    rename = {}
    if "lon" in elevation_source.coords:
        rename["lon"] = "longitude"
    if "lat" in elevation_source.coords:
        rename["lat"] = "latitude"

    elevation_source = elevation_source.rename(rename).sortby(["latitude", "longitude"])

    tid_rename = {}
    if "lon" in tid_source.coords:
        tid_rename["lon"] = "longitude"
    if "lat" in tid_source.coords:
        tid_rename["lat"] = "latitude"

    tid_source = tid_source.rename(tid_rename).sortby(["latitude", "longitude"])

    # Ensure both GEBCO files describe the same source grid.
    if not np.array_equal(
        elevation_source.latitude.values,
        tid_source.latitude.values,
    ):
        raise ValueError("GEBCO elevation and TID latitude grids do not match.")

    if not np.array_equal(
        elevation_source.longitude.values,
        tid_source.longitude.values,
    ):
        raise ValueError("GEBCO elevation and TID longitude grids do not match.")

    if "elevation" not in elevation_source:
        raise ValueError("GEBCO file does not contain 'elevation'.")

    if "tid" not in tid_source:
        raise ValueError("GEBCO TID file does not contain 'tid'.")

    # Elevation is continuous, so use linear interpolation.
    elevation = elevation_source["elevation"].interp(
        latitude=target_latitude,
        longitude=target_longitude,
        method="linear",
    )

    # TID represents categorical codes, so use nearest-neighbour interpolation.
    tid = tid_source["tid"].interp(
        latitude=target_latitude,
        longitude=target_longitude,
        method="nearest",
    )

    elevation = elevation.transpose("latitude", "longitude")
    tid = tid.transpose("latitude", "longitude")

    elevation_values = np.asarray(elevation.values, dtype=float)
    tid_values = np.asarray(tid.values, dtype=float)

    if not np.isfinite(elevation_values).all():
        raise ValueError("Interpolated GEBCO elevation contains invalid values.")

    if not np.isfinite(tid_values).all():
        raise ValueError("Interpolated GEBCO TID contains invalid values.")

    if not np.equal(tid_values, np.round(tid_values)).all():
        raise ValueError("Interpolated TID contains non-integer codes.")

    tid_values = np.round(tid_values).astype(np.int16)

    bathymetry = xr.Dataset(
        data_vars={
            "elevation_m": (
                ("latitude", "longitude"),
                elevation_values,
                {"units": "m"},
            ),
            "tid": (
                ("latitude", "longitude"),
                tid_values,
            ),
        },
        coords={
            "latitude": target_latitude,
            "longitude": target_longitude,
        },
        attrs={
            "provider": provenance["provider"],
            "source_dataset_id": provenance["dataset_id"],
            "source_dataset_version": provenance["dataset_version"],
            "source_version": provenance["source_version"],
            "license_name": provenance["license_name"],
            "source_file_path": str(GEBCO_FILE.relative_to(PROJECT_ROOT)),
            "tid_source_file_path": str(GEBCO_TID_FILE.relative_to(PROJECT_ROOT)),
            "vertical_reference": "unspecified",
            "geospatial_bounds_vertical_crs": "EPSG:5831",
        },
    )

    return normalize_gebco_bathymetry(bathymetry)


def main() -> None:
    print("Preparing offshore marine training dataset...")

    for path in (
        COPERNICUS_FILE,
        COPERNICUS_PROVENANCE,
        GEBCO_FILE,
        GEBCO_TID_FILE,
        GEBCO_PROVENANCE,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required source file not found: {path}")

    marine, latitude, longitude = prepare_copernicus_grid()

    print(f"Copernicus grid: {len(latitude)} latitude × {len(longitude)} longitude")

    bathymetry = prepare_gebco_grid(latitude, longitude)

    bathymetry_values = bathymetry["bathymetry_m"].values
    tid_values = bathymetry["tid"].values

    if np.any(bathymetry_values <= 0):
        raise ValueError("The selected grid contains land or invalid depths.")

    if not np.isfinite(bathymetry_values).all():
        raise ValueError("Bathymetry contains non-finite values.")

    print(
        f"Bathymetry: {bathymetry_values.shape}, "
        f"depth range: {bathymetry_values.min():.2f}–"
        f"{bathymetry_values.max():.2f} m"
    )

    print(f"TID codes: {dict(zip(*np.unique(tid_values, return_counts=True)))}")

    prepared = prepare_marine_dataset(marine, bathymetry)

    # Document the regional scope and preprocessing choices.
    prepared.metadata["domain"] = {
        "name": "Mombasa-adjacent offshore regional domain",
        "description": (
            "An offshore regional grid east of Mombasa. "
            "It does not cover Mombasa Port or its harbour."
        ),
        "west": float(longitude[0]),
        "east": float(longitude[-1]),
        "south": float(latitude[0]),
        "north": float(latitude[-1]),
        "nx": len(longitude),
        "ny": len(latitude),
        "crs": "EPSG:4326",
        "approximate_resolution_km": 9,
    }

    prepared.metadata["preprocessing"] = {
        "gebco_elevation_interpolation": "linear",
        "gebco_tid_interpolation": "nearest",
        "vertical_reference_note": (
            "Copernicus is normalized using the project's geoid convention. "
            "GEBCO vertical reference is unspecified in the source metadata; "
            "its vertical CRS is EPSG:5831. No vertical datum transformation "
            "has been applied."
        ),
        "operational_limitation": (
            "This regional dataset is not validated for harbour-scale "
            "predictions, navigation, safe-draft decisions, or operational "
            "scheduling."
        ),
    }

    prepared.metadata["source_provenance_files"] = {
        "copernicus": str(COPERNICUS_PROVENANCE.relative_to(PROJECT_ROOT)),
        "gebco": str(GEBCO_PROVENANCE.relative_to(PROJECT_ROOT)),
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    save_prepared_dataset(prepared, OUTPUT_FILE)

    # Reload the archive to verify that serialization preserved the dataset.
    reloaded = load_prepared_dataset(OUTPUT_FILE)

    if reloaded.metadata["grid"]["nx"] != len(longitude):
        raise ValueError("Reloaded dataset longitude dimension is incorrect.")

    if reloaded.metadata["grid"]["ny"] != len(latitude):
        raise ValueError("Reloaded dataset latitude dimension is incorrect.")

    if reloaded.split.total_samples != prepared.split.total_samples:
        raise ValueError("Reloaded sample count does not match.")

    print("\nDataset successfully generated and reloaded.")
    print(f"Output: {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")
    print(f"Generated samples: {reloaded.metadata['generated_samples']}")
    print(f"Training samples: {len(reloaded.split.train)}")
    print(f"Validation samples: {len(reloaded.split.validation)}")
    print(f"Test samples: {len(reloaded.split.test)}")
    print(f"Purged samples: {reloaded.metadata['purged_samples']}")


if __name__ == "__main__":
    main()
