from .dataset import generate_shallow_water_datasets
from .scenarios import (
    ShallowWaterScenario,
    get_fno_v2_scenarios,
    get_shallow_water_scenarios,
)

__all__ = [
    "ShallowWaterScenario",
    "generate_shallow_water_datasets",
    "get_fno_v2_scenarios",
    "get_shallow_water_scenarios",
]
