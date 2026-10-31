from dataclasses import dataclass

import xarray as xr

from .normalization import normalize_copernicus_dataset
from .provenance import MarineDataProvenance
from .sources.base import MarineDataRequest, MarineDataSource
from .validation import validate_marine_dataset


@dataclass
class MarineDataImportResult:
    """Contains an imported dataset and its provenance."""

    dataset: xr.Dataset
    provenance: MarineDataProvenance
    provenance_file_path: str


def validate_source_metadata(dataset: xr.Dataset) -> None:
    """Ensure the source dataset contains required provenance metadata."""

    required_metadata = (
        "provider",
        "source_dataset_id",
        "source_dataset_version",
        "source_version",
        "source_file_path",
    )

    missing_metadata = [
        field
        for field in required_metadata
        if not isinstance(dataset.attrs.get(field), str)
        or not dataset.attrs[field].strip()
        or dataset.attrs[field].strip().lower() == "unknown"
    ]

    if missing_metadata:
        raise ValueError(
            "Source dataset is missing required provenance metadata: "
            f"{missing_metadata}"
        )


def ingest_marine_data(
    source: MarineDataSource,
    request: MarineDataRequest,
    *,
    license_name: str,
) -> MarineDataImportResult:
    """
    Fetch, normalize, validate and record marine data provenance.
    """

    if not isinstance(license_name, str) or not license_name.strip():
        raise ValueError("A valid data license name is required.")

    # 1. Retrieve the original dataset.
    raw_dataset = source.fetch(request)

    try:
        # 2. Validate source metadata before processing.
        validate_source_metadata(raw_dataset)

        # 3. Normalize provider-specific variables and dimensions.
        normalized_dataset = normalize_copernicus_dataset(raw_dataset)

        # 4. Validate the normalized dataset.
        validate_marine_dataset(normalized_dataset)

        # 5. Capture source metadata.
        provenance = MarineDataProvenance.create(
            provider=raw_dataset.attrs["provider"],
            dataset_id=raw_dataset.attrs["source_dataset_id"],
            dataset_version=raw_dataset.attrs["source_dataset_version"],
            source_version=raw_dataset.attrs["source_version"],
            variables=request.variables,
            west=request.west,
            south=request.south,
            east=request.east,
            north=request.north,
            start_time=request.start_time,
            end_time=request.end_time,
            file_path=raw_dataset.attrs["source_file_path"],
            license_name=license_name,
        )

        provenance_file_path = provenance.save()

        return MarineDataImportResult(
            dataset=normalized_dataset,
            provenance=provenance,
            provenance_file_path=provenance_file_path,
        )

    finally:
        # Always release the original dataset resources.
        raw_dataset.close()
