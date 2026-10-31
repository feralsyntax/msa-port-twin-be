from dataclasses import asdict, dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class MarineDataProvenance:
    """Records the origin and retrieval details of a marine dataset."""

    provider: str
    dataset_id: str
    dataset_version: str
    source_version: str

    variables: tuple[str, ...]

    west: float
    south: float
    east: float
    north: float

    start_time: datetime
    end_time: datetime

    retrieved_at: datetime
    file_path: str
    file_format: str = "netcdf"
    license_name: str = "Copernicus Marine data license"

    @classmethod
    def create(
        cls,
        *,
        provider: str,
        dataset_id: str,
        dataset_version: str,
        source_version: str,
        variables: tuple[str, ...],
        west: float,
        south: float,
        east: float,
        north: float,
        start_time: datetime,
        end_time: datetime,
        file_path: str,
        license_name: str,
    ) -> "MarineDataProvenance":
        """Create provenance metadata with the current retrieval time."""

        return cls(
            provider=provider,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            source_version=source_version,
            variables=variables,
            west=west,
            south=south,
            east=east,
            north=north,
            start_time=start_time,
            end_time=end_time,
            retrieved_at=datetime.now(UTC),
            file_path=file_path,
            license_name=license_name,
        )

    def to_dict(self) -> dict:
        """Return provenance metadata in a JSON-serializable format."""

        data = asdict(self)

        for field in ("start_time", "end_time", "retrieved_at"):
            data[field] = data[field].isoformat()

        data["variables"] = list(self.variables)

        return data
