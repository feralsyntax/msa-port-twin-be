"""Two-dimensional depth-averaged shallow-water solver.

Uses finite-volume Rusanov fluxes, hydrostatic reconstruction,
SSP-RK2 time integration, wind stress, bottom friction and Coriolis.

The east boundary receives idealized sinusoidal tidal forcing.
Other external boundaries are reflective walls.
"""

from dataclasses import asdict

import numpy as np
import xarray as xr

from port_twin_app.services.marine_data.shallow_water.config import (
    ShallowWaterConfig,
)
from port_twin_app.services.marine_data.shallow_water.domain import (
    ShallowWaterDomain,
)


def _face_flux(
    eta_l,
    hu_l,
    hv_l,
    bed_l,
    wet_l,
    eta_r,
    hu_r,
    hv_r,
    bed_r,
    wet_r,
    gravity,
    normal,
):
    """Calculate hydrostatically reconstructed Rusanov face fluxes."""

    # Treat a wet/dry interface as a reflective wall.
    # This prevents water and momentum from flowing into land cells.
    left_only = wet_l & ~wet_r
    right_only = ~wet_l & wet_r
    neither = ~wet_l & ~wet_r

    bed_r = np.where(left_only, bed_l, bed_r)
    eta_r = np.where(left_only, eta_l, eta_r)
    hu_r = np.where(left_only, -hu_l if normal == "x" else hu_l, hu_r)
    hv_r = np.where(left_only, hv_l if normal == "x" else -hv_l, hv_r)

    bed_l = np.where(right_only, bed_r, bed_l)
    eta_l = np.where(right_only, eta_r, eta_l)
    hu_l = np.where(right_only, -hu_r if normal == "x" else hu_r, hu_l)
    hv_l = np.where(right_only, hv_r if normal == "x" else -hv_r, hv_l)

    depth_l = np.maximum(eta_l + bed_l, 0.0)
    depth_r = np.maximum(eta_r + bed_r, 0.0)

    # Reconstruct both sides at the higher bed elevation.
    bed_face = np.minimum(bed_l, bed_r)

    reconstructed_l = np.maximum(eta_l + bed_face, 0.0)
    reconstructed_r = np.maximum(eta_r + bed_face, 0.0)

    scale_l = np.divide(
        reconstructed_l,
        depth_l,
        out=np.zeros_like(reconstructed_l),
        where=depth_l > 0,
    )
    scale_r = np.divide(
        reconstructed_r,
        depth_r,
        out=np.zeros_like(reconstructed_r),
        where=depth_r > 0,
    )

    hu_star_l = hu_l * scale_l
    hv_star_l = hv_l * scale_l
    hu_star_r = hu_r * scale_r
    hv_star_r = hv_r * scale_r

    u_l = np.divide(
        hu_star_l,
        reconstructed_l,
        out=np.zeros_like(hu_star_l),
        where=reconstructed_l > 0,
    )
    v_l = np.divide(
        hv_star_l,
        reconstructed_l,
        out=np.zeros_like(hv_star_l),
        where=reconstructed_l > 0,
    )
    u_r = np.divide(
        hu_star_r,
        reconstructed_r,
        out=np.zeros_like(hu_star_r),
        where=reconstructed_r > 0,
    )
    v_r = np.divide(
        hv_star_r,
        reconstructed_r,
        out=np.zeros_like(hv_star_r),
        where=reconstructed_r > 0,
    )

    if normal == "x":
        normal_l, normal_r = u_l, u_r

        flux_l = np.stack(
            (
                hu_star_l,
                hu_star_l * u_l + 0.5 * gravity * reconstructed_l**2,
                hu_star_l * v_l,
            )
        )
        flux_r = np.stack(
            (
                hu_star_r,
                hu_star_r * u_r + 0.5 * gravity * reconstructed_r**2,
                hu_star_r * v_r,
            )
        )
        state_l = np.stack((reconstructed_l, hu_star_l, hv_star_l))
        state_r = np.stack((reconstructed_r, hu_star_r, hv_star_r))
        normal_index = 1
    else:
        normal_l, normal_r = v_l, v_r

        flux_l = np.stack(
            (
                hv_star_l,
                hv_star_l * u_l,
                hv_star_l * v_l + 0.5 * gravity * reconstructed_l**2,
            )
        )
        flux_r = np.stack(
            (
                hv_star_r,
                hv_star_r * u_r,
                hv_star_r * v_r + 0.5 * gravity * reconstructed_r**2,
            )
        )
        state_l = np.stack((reconstructed_l, hu_star_l, hv_star_l))
        state_r = np.stack((reconstructed_r, hu_star_r, hv_star_r))
        normal_index = 2

    speed = np.maximum(
        np.abs(normal_l) + np.sqrt(gravity * reconstructed_l),
        np.abs(normal_r) + np.sqrt(gravity * reconstructed_r),
    )

    common_flux = 0.5 * (flux_l + flux_r) - 0.5 * speed[np.newaxis, ...] * (
        state_r - state_l
    )

    # Side-specific hydrostatic pressure corrections.
    left_flux = common_flux.copy()
    right_flux = common_flux.copy()

    left_flux[normal_index] += 0.5 * gravity * (depth_l**2 - reconstructed_l**2)
    right_flux[normal_index] += 0.5 * gravity * (depth_r**2 - reconstructed_r**2)

    left_flux[:, neither] = 0.0
    right_flux[:, neither] = 0.0

    return left_flux, right_flux


def _calculate_rhs(
    eta,
    hu,
    hv,
    bed,
    water_mask,
    dx,
    dy,
    time_seconds,
    config,
):
    """Calculate finite-volume tendencies and physical source terms.

    Returns:
        Tuple containing water-level tendency, eastward momentum
        tendency, northward momentum tendency, and net boundary
        water inflow rate in cubic metres per second.
    """

    gravity = config.gravity_ms2

    depth = np.where(
        water_mask,
        np.maximum(eta + bed, 0.0),
        0.0,
    )

    dry = depth < config.minimum_water_depth_m
    depth[dry] = 0.0

    hu = np.where(dry, 0.0, hu)
    hv = np.where(dry, 0.0, hv)

    u = np.divide(
        hu,
        depth,
        out=np.zeros_like(hu),
        where=depth > 0,
    )
    v = np.divide(
        hv,
        depth,
        out=np.zeros_like(hv),
        where=depth > 0,
    )

    # West and east ghost cells.
    west_eta = eta[:, :1]
    west_hu = -hu[:, :1]
    west_hv = hv[:, :1]
    west_bed = bed[:, :1]
    west_wet = water_mask[:, :1]

    tide = config.tide_amplitude_m * np.sin(
        2.0 * np.pi * time_seconds / config.tide_period_seconds
        + config.tide_phase_radians
    )

    east_eta = np.full_like(eta[:, -1:], tide)
    east_bed = bed[:, -1:]
    east_depth = np.maximum(east_eta + east_bed, 0.0)

    east_hu = east_depth * u[:, -1:]
    east_hv = east_depth * v[:, -1:]
    east_wet = water_mask[:, -1:] & (east_depth >= config.minimum_water_depth_m)

    eta_x = np.concatenate((west_eta, eta, east_eta), axis=1)
    hu_x = np.concatenate((west_hu, hu, east_hu), axis=1)
    hv_x = np.concatenate((west_hv, hv, east_hv), axis=1)
    bed_x = np.concatenate((west_bed, bed, east_bed), axis=1)
    wet_x = np.concatenate((west_wet, water_mask, east_wet), axis=1)

    flux_x_left, flux_x_right = _face_flux(
        eta_x[:, :-1],
        hu_x[:, :-1],
        hv_x[:, :-1],
        bed_x[:, :-1],
        wet_x[:, :-1],
        eta_x[:, 1:],
        hu_x[:, 1:],
        hv_x[:, 1:],
        bed_x[:, 1:],
        wet_x[:, 1:],
        gravity,
        "x",
    )

    # South and north ghost cells are reflective.
    south_eta = eta[:1, :]
    south_hu = hu[:1, :]
    south_hv = -hv[:1, :]
    south_bed = bed[:1, :]
    south_wet = water_mask[:1, :]

    north_eta = eta[-1:, :]
    north_hu = hu[-1:, :]
    north_hv = -hv[-1:, :]
    north_bed = bed[-1:, :]
    north_wet = water_mask[-1:, :]

    eta_y = np.concatenate((south_eta, eta, north_eta), axis=0)
    hu_y = np.concatenate((south_hu, hu, north_hu), axis=0)
    hv_y = np.concatenate((south_hv, hv, north_hv), axis=0)
    bed_y = np.concatenate((south_bed, bed, north_bed), axis=0)
    wet_y = np.concatenate((south_wet, water_mask, north_wet), axis=0)

    flux_y_lower, flux_y_upper = _face_flux(
        eta_y[:-1, :],
        hu_y[:-1, :],
        hv_y[:-1, :],
        bed_y[:-1, :],
        wet_y[:-1, :],
        eta_y[1:, :],
        hu_y[1:, :],
        hv_y[1:, :],
        bed_y[1:, :],
        wet_y[1:, :],
        gravity,
        "y",
    )

    # Calculate net water entering through the four external boundaries.
    # Fluxes are in m²/s; multiplying by face length gives m³/s.
    east_outflow = np.sum(flux_x_left[0, :, -1]) * dy
    west_outflow = -np.sum(flux_x_right[0, :, 0]) * dy
    north_outflow = np.sum(flux_y_lower[0, -1, :]) * dx
    south_outflow = -np.sum(flux_y_upper[0, 0, :]) * dx

    boundary_inflow_rate = -(
        east_outflow + west_outflow + north_outflow + south_outflow
    )

    tendency = -(
        (flux_x_left[:, :, 1:] - flux_x_right[:, :, :-1]) / dx
        + (flux_y_lower[:, 1:, :] - flux_y_upper[:, :-1, :]) / dy
    )

    deta = tendency[0]
    dhu = tendency[1]
    dhv = tendency[2]

    # Wind stress transferred from air to the water surface.
    wind_speed = np.hypot(config.wind_u_ms, config.wind_v_ms)

    wind_stress_u = (
        config.air_density_kgm3
        * config.wind_drag_coefficient
        * wind_speed
        * config.wind_u_ms
        / config.water_density_kgm3
    )
    wind_stress_v = (
        config.air_density_kgm3
        * config.wind_drag_coefficient
        * wind_speed
        * config.wind_v_ms
        / config.water_density_kgm3
    )

    dhu += np.where(depth > 0, wind_stress_u, 0.0)
    dhv += np.where(depth > 0, wind_stress_v, 0.0)

    # Quadratic bottom friction.
    speed = np.hypot(u, v)
    dhu -= config.bottom_drag_coefficient * speed * u
    dhv -= config.bottom_drag_coefficient * speed * v

    # Northern-hemisphere convention is positive f; Mombasa is southern.
    dhu += config.coriolis_parameter_s1 * hv
    dhv -= config.coriolis_parameter_s1 * hu

    dhu = np.where(water_mask, dhu, 0.0)
    dhv = np.where(water_mask, dhv, 0.0)
    deta = np.where(water_mask, deta, 0.0)

    return deta, dhu, dhv, boundary_inflow_rate


def _stable_timestep(eta, hu, hv, bed, mask, dx, dy, config):
    """Return an adaptive timestep constrained by the 2D CFL condition."""

    depth = np.where(mask, np.maximum(eta + bed, 0.0), 0.0)
    wet = mask & (depth >= config.minimum_water_depth_m)

    u = np.divide(
        hu,
        depth,
        out=np.zeros_like(hu),
        where=wet,
    )
    v = np.divide(
        hv,
        depth,
        out=np.zeros_like(hv),
        where=wet,
    )

    wave_speed = np.sqrt(config.gravity_ms2 * depth)

    # Combine propagation rates from both spatial directions.
    cfl_rate = (np.abs(u) + wave_speed) / dx + (np.abs(v) + wave_speed) / dy

    maximum_rate = float(np.max(cfl_rate))

    if maximum_rate <= 0:
        return config.output_interval_seconds

    return config.cfl_number / maximum_rate


def _calculate_water_volume(eta, bed, mask, dx, dy):
    """Calculate total water volume in the computational domain."""

    depth = np.where(
        mask,
        np.maximum(eta + bed, 0.0),
        0.0,
    )

    return float(np.sum(depth, dtype=np.float64) * dx * dy)


def run_shallow_water_simulation(
    domain: ShallowWaterDomain,
    config: ShallowWaterConfig | None = None,
) -> xr.Dataset:
    """Run an idealized 2D shallow-water simulation.

    Returns hourly or configured-interval snapshots. The output is
    explicitly labelled simulated and is not observational ground truth.
    """

    config = config or ShallowWaterConfig()

    bed = np.asarray(domain.bathymetry_m, dtype=np.float64)
    mask = np.asarray(domain.water_mask, dtype=bool)

    if bed.ndim != 2 or mask.shape != bed.shape:
        raise ValueError("Domain bathymetry and water mask must be matching 2D arrays.")

    if not mask.any():
        raise ValueError("Domain contains no computational water cells.")

    if not np.isfinite(bed[mask]).all() or np.any(bed[mask] <= 0):
        raise ValueError("Computational bathymetry must be finite and positive down.")

    dx = domain.grid_spacing_m
    dy = domain.grid_spacing_m

    if dx <= 0 or dy <= 0:
        raise ValueError("Grid spacing must be positive.")

    ny, nx = bed.shape

    # Initial sea surface is at the assumed mean-sea-level reference.
    eta = np.zeros((ny, nx), dtype=np.float64)
    hu = np.zeros_like(eta)
    hv = np.zeros_like(eta)

    eta[~mask] = np.nan

    # State arrays use zero values internally for land cells.
    eta_state = np.where(mask, eta, 0.0)

    output_times = [0.0]
    water_levels = [eta.copy()]
    eastward_currents = [np.where(mask, 0.0, np.nan)]
    northward_currents = [np.where(mask, 0.0, np.nan)]

    # Track the water-volume budget from the initial state.
    initial_volume = _calculate_water_volume(
        eta_state,
        bed,
        mask,
        dx,
        dy,
    )

    water_volumes = [initial_volume]
    boundary_volume_exchange = 0.0
    boundary_volume_exchanges = [boundary_volume_exchange]

    time_seconds = 0.0
    next_output = min(
        config.output_interval_seconds,
        config.duration_seconds,
    )

    while time_seconds < config.duration_seconds:
        dt = min(
            _stable_timestep(
                eta_state,
                hu,
                hv,
                bed,
                mask,
                dx,
                dy,
                config,
            ),
            next_output - time_seconds,
            config.duration_seconds - time_seconds,
        )

        if not np.isfinite(dt) or dt <= 0:
            raise RuntimeError("Solver produced an invalid timestep.")

        # SSP-RK2 first stage.
        (
            deta_1,
            dhu_1,
            dhv_1,
            boundary_rate_1,
        ) = _calculate_rhs(
            eta_state,
            hu,
            hv,
            bed,
            mask,
            dx,
            dy,
            time_seconds,
            config,
        )

        eta_1 = eta_state + dt * deta_1
        hu_1 = hu + dt * dhu_1
        hv_1 = hv + dt * dhv_1

        # SSP-RK2 second stage.
        (
            deta_2,
            dhu_2,
            dhv_2,
            boundary_rate_2,
        ) = _calculate_rhs(
            eta_1,
            hu_1,
            hv_1,
            bed,
            mask,
            dx,
            dy,
            time_seconds + dt,
            config,
        )

        eta_state = 0.5 * (eta_state + eta_1 + dt * deta_2)
        hu = 0.5 * (hu + hu_1 + dt * dhu_2)
        hv = 0.5 * (hv + hv_1 + dt * dhv_2)

        # Integrate boundary exchange with the same RK2 weighting.
        boundary_volume_exchange += 0.5 * dt * (boundary_rate_1 + boundary_rate_2)

        # Enforce the wet/dry threshold and remove land momentum.
        depth = np.maximum(eta_state + bed, 0.0)
        dry = (~mask) | (depth < config.minimum_water_depth_m)

        hu[dry] = 0.0
        hv[dry] = 0.0

        time_seconds += dt

        if not (
            np.isfinite(eta_state[mask]).all()
            and np.isfinite(hu[mask]).all()
            and np.isfinite(hv[mask]).all()
        ):
            raise FloatingPointError(
                f"Non-finite solver state at t={time_seconds:.3f}s."
            )

        if time_seconds >= next_output - 1e-8:
            depth = np.maximum(eta_state + bed, 0.0)

            u = np.divide(
                hu,
                depth,
                out=np.zeros_like(hu),
                where=depth >= config.minimum_water_depth_m,
            )
            v = np.divide(
                hv,
                depth,
                out=np.zeros_like(hv),
                where=depth >= config.minimum_water_depth_m,
            )

            level_snapshot = np.where(mask, eta_state, np.nan)
            u_snapshot = np.where(mask, u, np.nan)
            v_snapshot = np.where(mask, v, np.nan)

            output_times.append(time_seconds)
            water_levels.append(level_snapshot.copy())
            eastward_currents.append(u_snapshot.copy())
            northward_currents.append(v_snapshot.copy())

            water_volumes.append(
                _calculate_water_volume(
                    eta_state,
                    bed,
                    mask,
                    dx,
                    dy,
                )
            )
            boundary_volume_exchanges.append(boundary_volume_exchange)

            next_output = min(
                next_output + config.output_interval_seconds,
                config.duration_seconds,
            )

    return xr.Dataset(
        data_vars={
            "water_level_m": (
                ("time_seconds", "northing", "easting"),
                np.asarray(water_levels, dtype=np.float32),
                {"units": "m", "positive": "up"},
            ),
            "current_u_ms": (
                ("time_seconds", "northing", "easting"),
                np.asarray(eastward_currents, dtype=np.float32),
                {
                    "units": "m s-1",
                    "long_name": "Eastward depth-averaged current",
                },
            ),
            "current_v_ms": (
                ("time_seconds", "northing", "easting"),
                np.asarray(northward_currents, dtype=np.float32),
                {
                    "units": "m s-1",
                    "long_name": "Northward depth-averaged current",
                },
            ),
            "bathymetry_m": (
                ("northing", "easting"),
                bed.astype(np.float32),
                {"units": "m", "positive": "down"},
            ),
            "water_mask": (
                ("northing", "easting"),
                mask,
            ),
            "water_volume_m3": (
                ("time_seconds",),
                np.asarray(water_volumes, dtype=np.float64),
                {
                    "units": "m3",
                    "long_name": "Total water volume in computational domain",
                },
            ),
            "boundary_volume_exchange_m3": (
                ("time_seconds",),
                np.asarray(boundary_volume_exchanges, dtype=np.float64),
                {
                    "units": "m3",
                    "long_name": (
                        "Cumulative net water entering through domain boundaries"
                    ),
                    "positive": "inflow",
                },
            ),
        },
        coords={
            "time_seconds": np.asarray(output_times, dtype=np.float64),
            "easting": domain.easting_m,
            "northing": domain.northing_m,
        },
        attrs={
            "data_classification": "simulated",
            "is_observed": False,
            "is_ground_truth": False,
            "solver": "2D nonlinear shallow-water equations",
            "numerical_flux": "Rusanov with hydrostatic reconstruction",
            "time_integrator": "SSP-RK2",
            "tidal_boundary": config.tidal_boundary,
            "other_boundaries": "reflective walls",
            "initial_condition": (
                "zero water-level anomaly relative to assumed MSL reference; "
                "zero initial currents"
            ),
            "vertical_reference": domain.source_vertical_reference,
            "bathymetry_source_version": domain.source_dataset_version,
            "bathymetry_sha256": domain.source_sha256,
            "grid_crs": domain.projected_crs,
            "grid_spacing_m": domain.grid_spacing_m,
            "random_seed": config.random_seed,
            "configuration": str(asdict(config)),
        },
    )
