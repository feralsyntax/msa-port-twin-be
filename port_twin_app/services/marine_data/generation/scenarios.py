from dataclasses import dataclass

from port_twin_app.services.marine_data.shallow_water.config import (
    ShallowWaterConfig,
)


@dataclass(frozen=True)
class ShallowWaterScenario:
    """A named, reproducible shallow-water forcing scenario."""

    name: str
    description: str
    config: ShallowWaterConfig


# All scenarios use the same seed because the current solver is
# deterministic and does not yet use stochastic forcing.
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


def get_shallow_water_scenarios(
    names: tuple[str, ...] | None = None,
) -> tuple[ShallowWaterScenario, ...]:
    """Return all scenarios, or a selected subset in the requested order."""

    if names is None:
        return _SCENARIOS

    scenarios_by_name = {scenario.name: scenario for scenario in _SCENARIOS}

    unknown_names = set(names) - scenarios_by_name.keys()
    if unknown_names:
        raise ValueError(
            f"Unknown shallow-water scenarios: {', '.join(sorted(unknown_names))}"
        )

    return tuple(scenarios_by_name[name] for name in names)
