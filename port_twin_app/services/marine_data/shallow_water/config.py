from dataclasses import dataclass


@dataclass(frozen=True)
class ShallowWaterConfig:
    """Reproducible configuration for idealized shallow-water simulations."""

    gravity_ms2: float = 9.81
    cfl_number: float = 0.4
    minimum_water_depth_m: float = 0.05

    bottom_drag_coefficient: float = 0.0025

    # Idealized wind forcing. Components are eastward and northward.
    wind_u_ms: float = 5.0
    wind_v_ms: float = 0.0
    air_density_kgm3: float = 1.225
    water_density_kgm3: float = 1025.0
    wind_drag_coefficient: float = 1.3e-3

    # Idealized M2-like tidal forcing, not locally calibrated.
    tide_amplitude_m: float = 0.5
    tide_period_seconds: float = 12.42 * 3600
    tide_phase_radians: float = 0.0
    tidal_boundary: str = "east"

    # Simulation and output settings.
    duration_seconds: float = 24 * 3600
    output_interval_seconds: float = 3600
    random_seed: int = 42

    def __post_init__(self):
        if self.gravity_ms2 <= 0:
            raise ValueError("Gravity must be positive.")

        if not 0 < self.cfl_number <= 1:
            raise ValueError("CFL number must be in (0, 1].")

        if self.minimum_water_depth_m <= 0:
            raise ValueError("Minimum water depth must be positive.")

        if self.bottom_drag_coefficient < 0:
            raise ValueError("Bottom drag cannot be negative.")

        if self.air_density_kgm3 <= 0 or self.water_density_kgm3 <= 0:
            raise ValueError("Fluid densities must be positive.")

        if self.wind_drag_coefficient < 0:
            raise ValueError("Wind drag cannot be negative.")

        if self.tide_amplitude_m < 0 or self.tide_period_seconds <= 0:
            raise ValueError("Tidal amplitude and period are invalid.")

        if self.tidal_boundary != "east":
            raise ValueError("Only the east tidal boundary is configured.")

        if self.duration_seconds <= 0 or self.output_interval_seconds <= 0:
            raise ValueError("Simulation and output durations must be positive.")

        if self.random_seed < 0:
            raise ValueError("Random seed cannot be negative.")
