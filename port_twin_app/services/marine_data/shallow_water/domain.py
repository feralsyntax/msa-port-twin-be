import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import label

DEFAULT_BATHYMETRY_PATH = (
    Path(__file__).resolve().parents[4]
    / "data"
    / "processed"
    / "marine"
    / "bathymetry"
    / "projected"
    / "approaches_gebco_2026_utm37s_250m.nc"
)


@dataclass(frozen=True)
class ShallowWaterDomain:
    """Projected bathymetry and connected computational water domain."""

    easting_m: np.ndarray
    northing_m: np.ndarray
    bathymetry_m: np.ndarray
    water_mask: np.ndarray
    grid_spacing_m: float
    projected_crs: str
    source_vertical_reference: str
    source_dataset_version: str
    source_file: str
    source_sha256: str
    total_water_cells: int
    excluded_water_cells: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_shallow_water_domain(
    path: str | Path = DEFAULT_BATHYMETRY_PATH,
) -> ShallowWaterDomain:
    """Load projected bathymetry and retain its largest connected water region."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Projected bathymetry not found: {path}")

    with xr.open_dataset(path) as source:
        dataset = source.load()

    required = {"bathymetry_m", "source_coverage"}

    if missing := required - set(dataset.data_vars):
        raise ValueError(f"Bathymetry dataset missing variables: {sorted(missing)}")

    if dataset.bathymetry_m.dims != ("northing", "easting"):
        raise ValueError("Bathymetry dimensions must be northing/easting.")

    if dataset.source_coverage.dims != ("northing", "easting"):
        raise ValueError("Source coverage dimensions must be northing/easting.")

    crs = dataset.attrs.get("projected_crs")

    if crs != "EPSG:32737":
        raise ValueError(f"Expected EPSG:32737, received {crs!r}.")

    easting = dataset.easting.values.astype(np.float64)
    northing = dataset.northing.values.astype(np.float64)

    if easting.size < 2 or northing.size < 2:
        raise ValueError("Bathymetry grid must have at least two cells per axis.")

    dx = np.diff(easting)
    dy = np.diff(northing)

    if (
        not np.isfinite(easting).all()
        or not np.isfinite(northing).all()
        or np.any(dx <= 0)
        or np.any(dy <= 0)
        or not np.allclose(dx, dx[0])
        or not np.allclose(dy, dy[0])
        or not np.isclose(dx[0], dy[0])
    ):
        raise ValueError("Expected uniformly spaced square grid cells.")

    if not np.isclose(
        dx[0],
        float(dataset.attrs.get("grid_resolution_m", np.nan)),
    ):
        raise ValueError("Grid spacing does not match declared resolution.")

    depth = dataset.bathymetry_m.values.astype(np.float64)
    coverage = dataset.source_coverage.values.astype(bool)

    if np.isinf(depth).any():
        raise ValueError("Bathymetry contains infinite values.")

    valid_water = coverage & np.isfinite(depth) & (depth > 0)

    if not valid_water.any():
        raise ValueError("Bathymetry contains no valid underwater cells.")

    # Four-neighbour connectivity avoids connecting diagonally touching cells.
    components, component_count = label(
        valid_water,
        structure=np.array(
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            dtype=np.uint8,
        ),
    )

    if component_count == 0:
        raise ValueError("No connected underwater region was found.")

    component_sizes = np.bincount(components.ravel())
    component_sizes[0] = 0
    main_component = int(np.argmax(component_sizes))

    water_mask = components == main_component
    selected_depth = np.where(water_mask, depth, np.nan)

    return ShallowWaterDomain(
        easting_m=easting,
        northing_m=northing,
        bathymetry_m=selected_depth,
        water_mask=water_mask,
        grid_spacing_m=float(dx[0]),
        projected_crs=crs,
        source_vertical_reference=str(
            dataset.attrs.get("source_vertical_reference", "unspecified")
        ),
        source_dataset_version=str(dataset.attrs.get("dataset_version", "unspecified")),
        source_file=str(path.resolve()),
        source_sha256=_sha256(path),
        total_water_cells=int(valid_water.sum()),
        excluded_water_cells=int(valid_water.sum() - water_mask.sum()),
    )
