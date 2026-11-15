import json
from decimal import Decimal
from pathlib import Path

import xarray as xr
from django.contrib.gis.geos import Polygon
from django.db import transaction

from port_twin_app.models import BathymetryDataset
from port_twin_app.services.marine_data.domains.mombasa_channel import (
    ZONES,
    get_zone,
)
from port_twin_app.services.marine_data.preparation.mombasa_bathymetry import (
    DEFAULT_OUTPUT_DIR,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _project_relative_path(path):
    """Return a repository-relative path and reject files outside the project."""
    resolved = Path(path).resolve()

    if not resolved.is_file():
        raise FileNotFoundError(f"Bathymetry file does not exist: {resolved}")

    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        raise ValueError(
            f"Bathymetry files must be inside the project directory: {resolved}"
        ) from None


def _decimal(value, field_name):
    try:
        result = Decimal(str(value))
    except Exception:  # noqa: BLE001
        raise ValueError(f"Invalid numeric value for {field_name}.") from None

    if not result.is_finite():
        raise ValueError(f"{field_name} must be finite.")

    return result


def _build_footprint(bounds):
    """Build a WGS84 polygon from actual dataset bounds."""
    try:
        west = float(bounds["west"])
        south = float(bounds["south"])
        east = float(bounds["east"])
        north = float(bounds["north"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("Invalid actual_data_bounds in provenance.") from None

    if not all(
        map(  # noqa: C417
            lambda value: value == value and abs(value) != float("inf"),  # noqa: PLR0124
            (west, south, east, north),
        )
    ):
        raise ValueError("Dataset footprint coordinates must be finite.")

    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("Dataset footprint bounds are invalid.")

    return Polygon(
        (
            (west, south),
            (east, south),
            (east, north),
            (west, north),
            (west, south),
        ),
        srid=4326,
    )


def register_mombasa_bathymetry(
    zone_name,
    output_dir=None,
):
    """Register or update a prepared Mombasa bathymetry dataset."""
    zone = get_zone(zone_name)
    output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)

    data_path = output_dir / f"{zone.name}_gebco_2026.nc"
    provenance_path = data_path.with_suffix(".provenance.json")

    data_relative_path = _project_relative_path(data_path)
    provenance_relative_path = _project_relative_path(provenance_path)

    with provenance_path.open(encoding="utf-8") as file:
        provenance = json.load(file)

    required = (
        "provider",
        "dataset_id",
        "dataset_version",
        "domain_name",
        "actual_data_bounds",
        "geographic_crs",
        "projected_crs",
        "target_resolution_m",
        "source_resolution",
        "variables",
        "vertical_reference",
        "bathymetry_positive_direction",
    )
    missing = [key for key in required if key not in provenance]
    if missing:
        raise ValueError(f"Bathymetry provenance is missing required fields: {missing}")

    if provenance["domain_name"] != zone.name:
        raise ValueError("Provenance domain does not match the requested zone.")

    if provenance["geographic_crs"] != "EPSG:4326":
        raise ValueError("Only EPSG:4326 bathymetry footprints are supported.")

    if provenance["provider"] != "GEBCO":
        raise ValueError("Unexpected bathymetry provider.")

    if not isinstance(provenance["variables"], list):
        raise ValueError("Provenance variables must be a list.")  # noqa: TRY004

    source_resolution = provenance["source_resolution"]
    longitude_spacing = _decimal(
        source_resolution["longitude"]["arc_seconds"],
        "longitude source spacing",
    )
    latitude_spacing = _decimal(
        source_resolution["latitude"]["arc_seconds"],
        "latitude source spacing",
    )

    with xr.open_dataset(data_path) as dataset:
        if not {"longitude", "latitude", "bathymetry_m"}.issubset(dataset.variables):
            raise ValueError("NetCDF is missing required bathymetry variables.")

        grid_width = dataset.sizes["longitude"]
        grid_height = dataset.sizes["latitude"]

        if not grid_width or not grid_height:
            raise ValueError("Bathymetry grid dimensions cannot be empty.")

        actual_variables = list(dataset.data_vars)

    if set(actual_variables) != set(provenance["variables"]):
        raise ValueError("NetCDF variables do not match the provenance metadata.")

    footprint = _build_footprint(provenance["actual_data_bounds"])

    correction = provenance.get("vertical_datum_correction_m")
    correction = (
        _decimal(correction, "vertical datum correction")
        if correction is not None
        else None
    )

    defaults = {
        "data_file_path": data_relative_path,
        "provenance_file_path": provenance_relative_path,
        "footprint": footprint,
        "geographic_crs": provenance["geographic_crs"],
        "projected_crs": provenance["projected_crs"],
        "target_resolution_m": int(provenance["target_resolution_m"]),
        "source_longitude_spacing_arc_seconds": longitude_spacing,
        "source_latitude_spacing_arc_seconds": latitude_spacing,
        "vertical_reference": provenance["vertical_reference"],
        "vertical_datum_correction_applied": bool(
            provenance.get("vertical_datum_correction_applied", False)
        ),
        "vertical_datum_correction_m": correction,
        "bathymetry_positive_direction": provenance["bathymetry_positive_direction"],
        "grid_width": grid_width,
        "grid_height": grid_height,
        "variables": provenance["variables"],
        "license": provenance.get("license", ""),
        "quality_status": BathymetryDataset.QualityStatus.PROVISIONAL,
        "provenance": provenance,
    }

    with transaction.atomic():
        record, created = BathymetryDataset.objects.update_or_create(
            provider=provenance["provider"],
            dataset_id=provenance["dataset_id"],
            dataset_version=provenance["dataset_version"],
            zone=zone.name,
            defaults=defaults,
        )

    return record, created


def register_all_mombasa_bathymetry(output_dir=None):
    """Register all configured Mombasa bathymetry zones."""
    return [
        register_mombasa_bathymetry(zone.name, output_dir=output_dir) for zone in ZONES
    ]
