import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from port_twin_app.services.marine_data.shallow_water.domain import (
    load_shallow_water_domain,
)
from port_twin_app.services.marine_data.shallow_water.solver import (
    run_shallow_water_simulation,
)

from .scenarios import get_shallow_water_scenarios

if TYPE_CHECKING:
    from port_twin_app.services.marine_data.shallow_water.domain import (
        ShallowWaterDomain,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "data" / "processed" / "marine" / "shallow_water"
)

SOLVER_VERSION = "1.0.0"


def _build_provenance(scenario, domain, dataset) -> dict:
    """Build provenance for one simulated shallow-water dataset."""

    return {
        "data_classification": "simulated",
        "is_observed": False,
        "is_ground_truth": False,
        "scenario": {
            "name": scenario.name,
            "description": scenario.description,
        },
        "solver": {
            "name": "2D shallow-water solver",
            "version": SOLVER_VERSION,
            "numerical_scheme": dataset.attrs.get("numerical_scheme"),
        },
        "configuration": asdict(scenario.config),
        "bathymetry": {
            "source_file": domain.source_file,
            "source_sha256": domain.source_sha256,
            "source_dataset_version": domain.source_dataset_version,
            "source_vertical_reference": domain.source_vertical_reference,
            "projected_crs": domain.projected_crs,
            "grid_spacing_m": domain.grid_spacing_m,
            "total_water_cells": domain.total_water_cells,
            "excluded_water_cells": domain.excluded_water_cells,
        },
        "output": {
            "format": "NetCDF",
            "engine": "h5netcdf",
            "dimensions": {name: int(size) for name, size in dataset.sizes.items()},
            "variables": list(dataset.data_vars),
        },
        "generated_at": datetime.now(UTC).isoformat(),
    }


def generate_shallow_water_datasets(
    scenario_names: tuple[str, ...] | None = None,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    duration_seconds: float | None = None,
    domain: "ShallowWaterDomain | None" = None,
) -> dict[str, Path]:
    """Generate and export selected simulated shallow-water scenarios.

    Returns a mapping of scenario names to their generated NetCDF paths.
    """

    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError("Duration must be positive.")

    scenarios = get_shallow_water_scenarios(scenario_names)
    if not scenarios:
        raise ValueError("At least one scenario must be selected.")

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    domain = domain or load_shallow_water_domain()
    generated_files = {}

    for scenario in scenarios:
        config = scenario.config
        if duration_seconds is not None:
            config = replace(config, duration_seconds=duration_seconds)

        selected_scenario = replace(scenario, config=config)
        dataset = run_shallow_water_simulation(domain, config)

        netcdf_path = output_directory / f"{scenario.name}.nc"
        provenance_path = output_directory / (f"{scenario.name}.provenance.json")

        dataset.attrs.update(
            {
                "scenario_name": scenario.name,
                "scenario_description": scenario.description,
                "data_classification": "simulated",
                "is_observed": False,
                "is_ground_truth": False,
            }
        )

        provenance = _build_provenance(
            selected_scenario,
            domain,
            dataset,
        )

        try:
            dataset.to_netcdf(netcdf_path, engine="h5netcdf")

            provenance_path.write_text(
                json.dumps(provenance, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        finally:
            dataset.close()

        generated_files[scenario.name] = netcdf_path

    return generated_files
