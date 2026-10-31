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


def ingest_marine_data(
    source: MarineDataSource,
    request: MarineDataRequest,
    *,
    license_name: str,
) -> MarineDataImportResult:
    """
    Fetch, normalize, validate and record marine data provenance.
    """

    if not license_name.strip():
        raise ValueError("A valid data license name is required.")

    # 1. Retrieve the original dataset.
    raw_dataset = source.fetch(request)

    try:
        # 2. Normalize provider-specific variables and dimensions.
        normalized_dataset = normalize_copernicus_dataset(raw_dataset)

        # 3. Validate the normalized dataset.
        validate_marine_dataset(normalized_dataset)

        # 4. Capture source metadata.
        provenance = MarineDataProvenance.create(
            provider=raw_dataset.attrs.get("provider", "unknown"),
            dataset_id=raw_dataset.attrs.get("source_dataset_id", "unknown"),
            dataset_version=raw_dataset.attrs.get("source_dataset_version", "unknown"),
            source_version=raw_dataset.attrs.get("source_version", "unknown"),
            variables=request.variables,
            west=request.west,
            south=request.south,
            east=request.east,
            north=request.north,
            start_time=request.start_time,
            end_time=request.end_time,
            file_path=raw_dataset.attrs.get("source_file_path", ""),
            license_name=license_name,
        )

        return MarineDataImportResult(
            dataset=normalized_dataset,
            provenance=provenance,
        )

    except Exception:
        raw_dataset.close()
        raise

    finally:
        if raw_dataset is not None:
            raw_dataset.close()
