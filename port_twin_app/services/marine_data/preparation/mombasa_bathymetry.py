import json
from pathlib import Path

import numpy as np

from port_twin_app.services.marine_data.domains.mombasa_channel import (
    GEOGRAPHIC_CRS,
    PROJECTED_CRS,
    get_zone,
)
from port_twin_app.services.marine_data.normalization import (
    normalize_gebco_bathymetry,
)
from port_twin_app.services.marine_data.sources.bathymetry import (
    BathymetryRequest,
)
from port_twin_app.services.marine_data.sources.gebco import (
    GEBCOBathymetrySource,
)
from port_twin_app.services.marine_data.validation import (
    validate_bathymetry_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_ELEVATION_PATH = PROJECT_ROOT / "data/raw/gebco/gebco_2026_mombasa_extended.nc"
DEFAULT_TID_PATH = PROJECT_ROOT / "data/raw/gebco/gebco_2026_mombasa_extended_tid.nc"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/marine/bathymetry"


def _coordinate_spacing(dataset, coordinate):
    """Return median coordinate spacing in degrees and arc seconds."""
    values = dataset[coordinate].values

    if len(values) < 2:
        raise ValueError(f"At least two {coordinate} coordinates are required.")

    spacing_degrees = float(np.median(np.abs(np.diff(values))))

    return {
        "degrees": spacing_degrees,
        "arc_seconds": spacing_degrees * 3600,
    }


def build_mombasa_bathymetry(zone_name, source=None):
    """Fetch, normalize and validate bathymetry for a configured zone."""
    zone = get_zone(zone_name)

    if source is None:
        source = GEBCOBathymetrySource(
            file_path=DEFAULT_ELEVATION_PATH,
            dataset_version="GEBCO_2026",
            tid_file_path=DEFAULT_TID_PATH,
        )

    request = BathymetryRequest(
        west=zone.bounds.west,
        south=zone.bounds.south,
        east=zone.bounds.east,
        north=zone.bounds.north,
    )

    raw_dataset = source.fetch(request)
    dataset = normalize_gebco_bathymetry(raw_dataset)

    validate_bathymetry_dataset(dataset)

    if "tid" not in dataset:
        raise ValueError(
            "GEBCO Type Identifier data is required for Mombasa bathymetry preparation."
        )

    if dataset["tid"].shape != dataset["bathymetry_m"].shape:
        raise ValueError("GEBCO TID grid does not match the bathymetry grid.")

    longitude_spacing = _coordinate_spacing(dataset, "longitude")
    latitude_spacing = _coordinate_spacing(dataset, "latitude")

    dataset.attrs.update(
        {
            "domain_name": zone.name,
            "domain_source_reference": zone.source_reference,
            "geographic_crs": GEOGRAPHIC_CRS,
            "projected_crs": PROJECTED_CRS,
            "target_resolution_m": zone.target_resolution_m,
            "source_resolution_longitude_degrees": (longitude_spacing["degrees"]),
            "source_resolution_longitude_arc_seconds": (
                longitude_spacing["arc_seconds"]
            ),
            "source_resolution_latitude_degrees": (latitude_spacing["degrees"]),
            "source_resolution_latitude_arc_seconds": (latitude_spacing["arc_seconds"]),
            "source_vertical_reference": raw_dataset.attrs.get(
                "vertical_reference",
                "unspecified",
            ),
            "vertical_datum_correction_applied": "false",
            "vertical_datum_correction_m": 0.0,
            "source_elevation_positive_direction": "up",
            "bathymetry_positive_direction": "down",
            "source_elevation_file": str(raw_dataset.attrs.get("source_file_path", "")),
            "source_tid_file": str(raw_dataset.attrs.get("tid_source_file_path", "")),
        }
    )

    return dataset


def prepare_mombasa_bathymetry(
    zone_name,
    output_dir=None,
    source=None,
):
    """Prepare and save a zone's NetCDF bathymetry and provenance."""
    zone = get_zone(zone_name)
    output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_mombasa_bathymetry(
        zone_name=zone_name,
        source=source,
    )

    output_path = output_dir / f"{zone.name}_gebco_2026.nc"
    provenance_path = output_path.with_suffix(".provenance.json")

    dataset.to_netcdf(output_path, engine="scipy")

    provenance = {
        "provider": "GEBCO",
        "dataset_id": "GEBCO",
        "dataset_version": "GEBCO_2026",
        "domain_name": zone.name,
        "domain_source_reference": zone.source_reference,
        "requested_bounds": {
            "west": zone.bounds.west,
            "south": zone.bounds.south,
            "east": zone.bounds.east,
            "north": zone.bounds.north,
        },
        "actual_data_bounds": {
            "west": float(dataset.longitude.min().item()),
            "south": float(dataset.latitude.min().item()),
            "east": float(dataset.longitude.max().item()),
            "north": float(dataset.latitude.max().item()),
        },
        "geographic_crs": GEOGRAPHIC_CRS,
        "projected_crs": PROJECTED_CRS,
        "target_resolution_m": zone.target_resolution_m,
        "source_resolution": {
            "longitude": _coordinate_spacing(dataset, "longitude"),
            "latitude": _coordinate_spacing(dataset, "latitude"),
        },
        "variables": list(dataset.data_vars),
        "source_elevation_file": dataset.attrs["source_elevation_file"],
        "source_tid_file": dataset.attrs["source_tid_file"],
        "vertical_reference": dataset.attrs["source_vertical_reference"],
        "vertical_datum_correction_applied": False,
        "vertical_datum_correction_m": None,
        "bathymetry_positive_direction": "down",
        "license": ("GEBCO public domain subject to its terms and disclaimer"),
        "output_file": str(output_path),
    }

    provenance_path.write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )

    return {
        "dataset": dataset,
        "output_path": output_path,
        "provenance_path": provenance_path,
    }
