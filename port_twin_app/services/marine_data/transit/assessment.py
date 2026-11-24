"""Experimental advisory transit-window assessment."""

import numpy as np

from port_twin_app.services.marine_data.generation.scenarios import (
    get_shallow_water_scenarios,
)
from port_twin_app.services.marine_data.shallow_water.domain import (
    DEFAULT_BATHYMETRY_PATH,
    load_shallow_water_domain,
)
from port_twin_app.services.marine_data.shallow_water.solver import (
    run_shallow_water_simulation,
)
from port_twin_app.services.marine_data.transit.config import (
    TransitAssessmentConfig,
)

DOMAIN_BATHYMETRY_PATHS = {
    "approaches": DEFAULT_BATHYMETRY_PATH,
    "port_operations": DEFAULT_BATHYMETRY_PATH.with_name(
        "port_operations_gebco_2026_utm37s_100m.nc"
    ),
}

ASSESSMENT_LIMITATIONS = [
    "Results are simulated scenario assessments, not observations or operational forecasts.",
    "Wind is constant scenario forcing, not a spatial wind field or forecast.",
    "The computational domain is rectangular and is not an actual navigational channel polygon.",
    "Clearance and current extrema are calculated across the connected computational water domain, not a validated vessel route.",
    "Bathymetry is a resampled computational product and is not established as navigation-grade depth data.",
    "No vertical datum correction has been applied to the bathymetry.",
    "Passing samples do not establish safe navigation or continuous conditions between model output times.",
    "This experimental assessment is not navigation or vessel-safety advice.",
]


def _load_domain(domain_name: str):
    """Load a supported computational bathymetry domain."""
    try:
        path = DOMAIN_BATHYMETRY_PATHS[domain_name]
    except KeyError as exc:
        supported = ", ".join(sorted(DOMAIN_BATHYMETRY_PATHS))
        raise ValueError(
            f"Unsupported domain '{domain_name}'. Choose from: {supported}."
        ) from exc

    return load_shallow_water_domain(path)


def _calculate_sample_metrics(dataset, draft_m: float) -> list[dict]:
    """Calculate domain-wide UKC and current extrema at each output time."""
    water_mask = dataset["water_mask"].values.astype(bool)
    bathymetry = dataset["bathymetry_m"].values.astype(np.float64)

    water_levels = dataset["water_level_m"].values.astype(np.float64)
    current_u = dataset["current_u_ms"].values.astype(np.float64)
    current_v = dataset["current_v_ms"].values.astype(np.float64)
    times = dataset["time_seconds"].values.astype(np.float64)

    if not water_mask.any():
        raise ValueError("Solver output contains no computational water cells.")

    if bathymetry.shape != water_mask.shape:
        raise ValueError("Solver bathymetry and water mask shapes do not match.")

    expected_shape = (times.size, *water_mask.shape)

    for name, values in (
        ("water level", water_levels),
        ("eastward current", current_u),
        ("northward current", current_v),
    ):
        if values.shape != expected_shape:
            raise ValueError(f"Solver {name} output has an unexpected shape.")

        if not np.isfinite(values[:, water_mask]).all():
            raise ValueError(f"Solver {name} output contains non-finite water values.")

    if not np.isfinite(bathymetry[water_mask]).all():
        raise ValueError("Solver bathymetry contains non-finite water values.")

    if (
        times.ndim != 1
        or times.size == 0
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("Solver output times must be finite and strictly increasing.")

    samples = []

    for index, time_seconds in enumerate(times):
        ukc = bathymetry[water_mask] + water_levels[index][water_mask] - draft_m

        current_speed = np.hypot(
            current_u[index][water_mask],
            current_v[index][water_mask],
        )

        samples.append(
            {
                "elapsed_seconds": float(time_seconds),
                "minimum_ukc_m": float(np.min(ukc)),
                "maximum_current_speed_ms": float(np.max(current_speed)),
            }
        )

    return samples


def _group_passing_samples(samples: list[dict]) -> list[dict]:
    """Group adjacent passing samples without implying continuous availability."""
    windows = []
    current_window = []

    def save_window():
        if not current_window:
            return

        sample_times = [sample["elapsed_seconds"] for sample in current_window]

        windows.append(
            {
                "start_elapsed_seconds": sample_times[0],
                "end_elapsed_seconds": sample_times[-1],
                "span_seconds": sample_times[-1] - sample_times[0],
                "sample_count": len(current_window),
                "sample_times_seconds": sample_times,
            }
        )

    for sample in samples:
        if sample["passes"]:
            current_window.append(sample)
        else:
            save_window()
            current_window = []

    save_window()
    return windows


def run_transit_assessment(
    scenario_name: str,
    domain_name: str = "approaches",
    assessment_config: TransitAssessmentConfig | None = None,
) -> dict:
    """Run a shallow-water scenario and calculate experimental transit windows."""
    assessment_config = assessment_config or TransitAssessmentConfig()

    scenarios = get_shallow_water_scenarios((scenario_name,))
    scenario = scenarios[0]

    domain = _load_domain(domain_name)
    dataset = run_shallow_water_simulation(domain, scenario.config)

    wind_speed = float(
        np.hypot(
            scenario.config.wind_u_ms,
            scenario.config.wind_v_ms,
        )
    )

    wind_passes = wind_speed <= assessment_config.maximum_wind_speed_ms

    draft_results = []

    for draft_class in assessment_config.draft_classes:
        samples = _calculate_sample_metrics(
            dataset,
            draft_class.representative_draft_m,
        )

        assessed_samples = []

        for sample in samples:
            ukc_passes = sample["minimum_ukc_m"] >= draft_class.minimum_ukc_m

            current_passes = (
                sample["maximum_current_speed_ms"]
                <= assessment_config.maximum_current_speed_ms
            )

            reasons = []

            if not ukc_passes:
                reasons.append("minimum_ukc_below_threshold")

            if not current_passes:
                reasons.append("current_above_threshold")

            if not wind_passes:
                reasons.append("wind_above_threshold")

            assessed_samples.append(
                {
                    **sample,
                    "wind_speed_ms": wind_speed,
                    "passes": not reasons,
                    "failure_reasons": reasons,
                }
            )

        draft_results.append(
            {
                "draft_class": draft_class.name,
                "representative_draft_m": draft_class.representative_draft_m,
                "minimum_ukc_required_m": draft_class.minimum_ukc_m,
                "windows": _group_passing_samples(assessed_samples),
                "samples": assessed_samples,
            }
        )

    return {
        "assessment_type": "experimental",
        "advisory_only": True,
        "data_classification": "simulated",
        "time_basis": "elapsed_seconds_from_simulation_start",
        "window_interpretation": (
            "Windows group consecutive passing solver output samples. "
            "They do not establish continuous conditions between samples."
        ),
        "scenario": {
            "name": scenario.name,
            "description": scenario.description,
            "dataset_version": scenario.dataset_version,
        },
        "domain": {
            "name": domain_name,
            "projected_crs": domain.projected_crs,
            "grid_spacing_m": domain.grid_spacing_m,
            "water_cell_count": domain.total_water_cells,
            "excluded_water_cell_count": domain.excluded_water_cells,
        },
        "forcing": {
            "wind_u_ms": scenario.config.wind_u_ms,
            "wind_v_ms": scenario.config.wind_v_ms,
            "wind_speed_ms": wind_speed,
            "wind_source": "constant_shallow_water_scenario_forcing",
            "tide_amplitude_m": scenario.config.tide_amplitude_m,
        },
        "thresholds": assessment_config.to_dict(),
        "solver_configuration": {
            "duration_seconds": scenario.config.duration_seconds,
            "output_interval_seconds": scenario.config.output_interval_seconds,
            "random_seed": scenario.config.random_seed,
        },
        "bathymetry_provenance": {
            "source_file": domain.source_file,
            "source_sha256": domain.source_sha256,
            "source_dataset_version": domain.source_dataset_version,
            "vertical_reference": domain.source_vertical_reference,
            "vertical_datum_correction_applied": False,
        },
        "draft_classes": draft_results,
        "limitations": list(ASSESSMENT_LIMITATIONS),
    }
