from dataclasses import dataclass
from itertools import product

from port_twin_app.services.marine_data.shallow_water.config import (
    ShallowWaterConfig,
)


@dataclass(frozen=True)
class ShallowWaterScenario:
    """A named, reproducible shallow-water forcing scenario."""

    name: str
    description: str
    config: ShallowWaterConfig
    dataset_version: str = "v1"


# Original BE-55 scenarios. Keep these unchanged for reproducibility.
_SCENARIOS = (
    ShallowWaterScenario(
        name="calm_tidal_baseline",
        description="Tidal forcing without wind.",
        config=ShallowWaterConfig(
            wind_u_ms=0.0,
            wind_v_ms=0.0,
            tide_amplitude_m=0.5,
            duration_seconds=6 * 3600,
            random_seed=42,
        ),
    ),
    ShallowWaterScenario(
        name="light_eastward_wind",
        description="Light wind forcing towards the east.",
        config=ShallowWaterConfig(
            wind_u_ms=3.0,
            wind_v_ms=0.0,
            tide_amplitude_m=0.5,
            duration_seconds=6 * 3600,
            random_seed=42,
        ),
    ),
    ShallowWaterScenario(
        name="moderate_eastward_wind",
        description="Moderate wind forcing towards the east.",
        config=ShallowWaterConfig(
            wind_u_ms=5.0,
            wind_v_ms=0.0,
            tide_amplitude_m=0.5,
            duration_seconds=6 * 3600,
            random_seed=42,
        ),
    ),
    ShallowWaterScenario(
        name="light_westward_wind",
        description="Light wind forcing towards the west.",
        config=ShallowWaterConfig(
            wind_u_ms=-3.0,
            wind_v_ms=0.0,
            tide_amplitude_m=0.5,
            duration_seconds=6 * 3600,
            random_seed=42,
        ),
    ),
    ShallowWaterScenario(
        name="light_northward_wind",
        description="Light wind forcing towards the north.",
        config=ShallowWaterConfig(
            wind_u_ms=0.0,
            wind_v_ms=3.0,
            tide_amplitude_m=0.5,
            duration_seconds=6 * 3600,
            random_seed=42,
        ),
    ),
    ShallowWaterScenario(
        name="stronger_tide",
        description="Stronger tidal forcing without wind.",
        config=ShallowWaterConfig(
            wind_u_ms=0.0,
            wind_v_ms=0.0,
            tide_amplitude_m=0.8,
            duration_seconds=6 * 3600,
            random_seed=42,
        ),
    ),
)


# Six wind profiles crossed with three tidal profiles produce 18 runs.
# Amplitude and phase are paired in this initial design; their effects
# cannot be isolated from one another using this scenario set alone.
_V2_WIND_PROFILES = (
    ("calm", 0.0, 0.0),
    ("eastward_3ms", 3.0, 0.0),
    ("westward_3ms", -3.0, 0.0),
    ("northward_3ms", 0.0, 3.0),
    ("southward_3ms", 0.0, -3.0),
    ("eastward_5ms", 5.0, 0.0),
)

_V2_TIDAL_PROFILES = (
    ("low_tide", 0.35, 0.0),
    ("nominal_tide", 0.5, 1.0471975511965976),
    ("high_tide", 0.8, 2.0943951023931953),
)


def get_shallow_water_scenarios(
    names: tuple[str, ...] | None = None,
) -> tuple[ShallowWaterScenario, ...]:
    """Return the original BE-55 scenarios, optionally selected by name."""

    if names is None:
        return _SCENARIOS

    scenarios_by_name = {scenario.name: scenario for scenario in _SCENARIOS}

    unknown_names = set(names) - scenarios_by_name.keys()
    if unknown_names:
        raise ValueError(
            f"Unknown shallow-water scenarios: {', '.join(sorted(unknown_names))}"
        )

    return tuple(scenarios_by_name[name] for name in names)


def get_fno_v2_scenarios(
    names: tuple[str, ...] | None = None,
) -> tuple[ShallowWaterScenario, ...]:
    """Return the expanded 18-run FNO v2 scenario set."""

    scenarios = []

    for (
        wind_profile,
        tide_profile,
    ) in product(_V2_WIND_PROFILES, _V2_TIDAL_PROFILES):
        wind_name, wind_u, wind_v = wind_profile
        tide_name, tide_amplitude, tide_phase = tide_profile

        scenarios.append(
            ShallowWaterScenario(
                name=f"v2_{wind_name}_{tide_name}",
                description=(
                    f"FNO v2 scenario with {wind_name.replace('_', ' ')} wind "
                    f"and {tide_name.replace('_', ' ')} forcing."
                ),
                config=ShallowWaterConfig(
                    wind_u_ms=wind_u,
                    wind_v_ms=wind_v,
                    tide_amplitude_m=tide_amplitude,
                    tide_phase_radians=tide_phase,
                    duration_seconds=12 * 3600,
                    random_seed=42,
                ),
                dataset_version="v2",
            )
        )

    if names is None:
        return tuple(scenarios)

    scenarios_by_name = {scenario.name: scenario for scenario in scenarios}
    unknown_names = set(names) - scenarios_by_name.keys()

    if unknown_names:
        raise ValueError(
            f"Unknown FNO v2 scenarios: {', '.join(sorted(unknown_names))}"
        )

    return tuple(scenarios_by_name[name] for name in names)
