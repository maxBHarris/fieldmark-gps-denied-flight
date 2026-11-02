"""Stateless REST API wrapping the fieldmark simulation core for the React
frontend. Every endpoint takes the full waypoints/params it needs and
rebuilds whatever it requires -- FieldRegion already disk-caches synthesized
grids by a hash of (lat0, lon0, half_width_km, n_grid, ...), so repeat calls
for the same region are near-instant after the first build.

Run with:
    uvicorn backend.main:app --reload
"""
import dataclasses
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from fieldmark.constants import LocalTangentPlane
from fieldmark.fieldmap import FieldRegion
from fieldmark.trajectory import from_waypoints
from fieldmark.imu import GRADES
from fieldmark import ekf, simulate

app = FastAPI(title="fieldmark API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev tool, not deployed -- fine to be permissive
    allow_methods=["*"],
    allow_headers=["*"],
)

MIN_HALF_WIDTH_KM = 30.0
# Capped mainly for interactive build latency (a bigger region means a
# bigger field-map grid to synthesize) and because the flat-Earth local
# tangent plane (fieldmark/constants.py:LocalTangentPlane) the whole
# simulation runs in accumulates distortion over very long distances.
MAX_HALF_WIDTH_KM = 600.0
PADDING_FRAC = 0.3
SWEEP_VALUES = [0.1, 0.3, 1, 3, 10, 30, 100]
SWEEP_2D_VALUES = [0.3, 1, 3, 10, 30]
DRIFT_HORIZONS_S = [30, 60, 120, 300, 600, 1200]


# ---- request/response models -------------------------------------------

class LatLon(BaseModel):
    lat: float
    lon: float


class SimulateRequest(BaseModel):
    waypoints: List[LatLon]
    imu_grade: str = "tactical"
    mag_noise_nt: float = 1.0
    grav_noise_mgal: float = 2.0
    use_mag: bool = True
    use_grav: bool = True
    speed_mps: float = 180.0
    turn_rate_dps: float = 3.0
    dt: float = 1.0
    seed: int = 0
    n_grid: int = 81
    max_points: int = 1500


class SweepRequest(BaseModel):
    waypoints: List[LatLon]
    imu_grade: str = "tactical"
    speed_mps: float = 180.0
    turn_rate_dps: float = 3.0
    dt: float = 1.0
    n_trials: int = 5
    n_grid: int = 81


class DriftRequest(BaseModel):
    waypoints: List[LatLon]
    speed_mps: float = 180.0
    turn_rate_dps: float = 3.0
    dt: float = 1.0
    n_trials: int = 5


# ---- shared region/trajectory construction ------------------------------

def _waypoints_to_trajectory(waypoints, speed_mps, turn_rate_dps, dt):
    """Convert lat/lon waypoints to a Trajectory, with no FieldRegion built --
    for endpoints (like /api/drift) that don't need the field maps at all."""
    if len(waypoints) < 2:
        raise HTTPException(400, "Need at least 2 waypoints")

    lats = [w.lat for w in waypoints]
    lons = [w.lon for w in waypoints]
    lat0 = sum(lats) / len(lats)
    lon0 = sum(lons) / len(lons)

    ltp = LocalTangentPlane(lat0, lon0)
    waypoints_en = [ltp.ll_to_en(w.lat, w.lon) for w in waypoints]
    return from_waypoints(waypoints_en, speed_mps=speed_mps,
                           turn_rate_dps=turn_rate_dps, dt=dt)


def _build_region_and_trajectory(waypoints, speed_mps, turn_rate_dps, dt, n_grid):
    if len(waypoints) < 2:
        raise HTTPException(400, "Need at least 2 waypoints")

    lats = [w.lat for w in waypoints]
    lons = [w.lon for w in waypoints]
    lat0 = sum(lats) / len(lats)
    lon0 = sum(lons) / len(lons)

    ltp = LocalTangentPlane(lat0, lon0)
    waypoints_en = [ltp.ll_to_en(w.lat, w.lon) for w in waypoints]
    half_extent_m = max(max(abs(e), abs(n)) for e, n in waypoints_en)

    # A coordinated turn "rounds the corner" rather than pivoting exactly on
    # a waypoint (see fieldmark/trajectory.py:_turn_then_leg), so the actual
    # flown path -- and the EKF's own position estimate, which needs some
    # room to wander during initial convergence -- can extend past the
    # waypoints' bounding box by roughly a turn radius. Sizing the region
    # only off the waypoints (no turn margin) was observed to let both the
    # true path and the estimate exit the map on tight/sharp routes, which
    # permanently cuts off map-matching updates for the rest of the flight
    # (FieldRegion.in_bounds gates measurement updates) -- so the margin
    # here is turn-geometry-aware, not just a fixed percentage.
    turn_radius_m = speed_mps / max(np.deg2rad(turn_rate_dps), 1e-6)
    margin_m = half_extent_m * PADDING_FRAC + 2.5 * turn_radius_m
    half_width_km = max(MIN_HALF_WIDTH_KM, (half_extent_m + margin_m) / 1000.0)
    if half_width_km > MAX_HALF_WIDTH_KM:
        raise HTTPException(
            400,
            f"Waypoints span too large an area (~{half_extent_m / 1000.0:.0f} km half-extent "
            f"from the route's center) -- keep the route within about "
            f"{MAX_HALF_WIDTH_KM:.0f} km of its center for this simulator.",
        )

    region = FieldRegion(lat0, lon0, half_width_km=half_width_km, n_grid=n_grid)
    trajectory = from_waypoints(waypoints_en, speed_mps=speed_mps,
                                 turn_rate_dps=turn_rate_dps, dt=dt)
    return region, trajectory


def _downsample(n_steps, max_points):
    if n_steps <= max_points:
        return np.arange(n_steps)
    return np.unique(np.linspace(0, n_steps - 1, max_points).astype(int))


def _region_payload(region):
    return {
        "lat0": region.lat0,
        "lon0": region.lon0,
        "half_width_km": region.half_width_km,
        "east_km": (region.east / 1000.0).tolist(),
        "north_km": (region.north / 1000.0).tolist(),
        "mag_grid": region.mag_grid.tolist(),
        "grav_grid": region.grav_grid.tolist(),
    }


# ---- endpoints ------------------------------------------------------------

@app.post("/api/simulate")
def simulate_flight(req: SimulateRequest):
    if req.imu_grade not in GRADES:
        raise HTTPException(400, f"Unknown imu_grade '{req.imu_grade}'")

    region, trajectory = _build_region_and_trajectory(
        req.waypoints, req.speed_mps, req.turn_rate_dps, req.dt, req.n_grid)
    grade = GRADES[req.imu_grade]
    cfg = ekf.EKFConfig(
        mag_noise_std_nt=req.mag_noise_nt,
        grav_noise_std_mgal=req.grav_noise_mgal,
        use_mag=req.use_mag,
        use_grav=req.use_grav,
    )
    result, summary = simulate.run_trial(region, trajectory, grade, cfg, dt=req.dt, seed=req.seed)

    idx = _downsample(len(result.t), req.max_points)
    true_lat, true_lon = region.ltp.en_to_ll(result.true_e[idx], result.true_n[idx])
    est_lat, est_lon = region.ltp.en_to_ll(result.est_e[idx], result.est_n[idx])

    return {
        "t": result.t[idx].tolist(),
        "true_lat": true_lat.tolist(),
        "true_lon": true_lon.tolist(),
        "est_lat": est_lat.tolist(),
        "est_lon": est_lon.tolist(),
        "pos_error_m": result.pos_error_m[idx].tolist(),
        "pos_std_m": result.pos_std_m[idx].tolist(),
        # Compass bearing (0=N, clockwise) of the EKF's estimated track at
        # each sample -- the frontend uses this to draw a corridor
        # perpendicular to the track (width = k*pos_std_m) along its whole
        # length, rather than a single uncertainty circle at one instant.
        "est_heading_deg": np.degrees(result.est_heading[idx]).tolist(),
        "mag_updates": result.mag_updates,
        "grav_updates": result.grav_updates,
        "tail_rms_m": summary.tail_rms_m,
        "final_error_m": summary.final_error_m,
        "max_error_m": summary.max_error_m,
        "region": _region_payload(region),
    }


@app.post("/api/sweep/mag")
def sweep_mag(req: SweepRequest):
    if req.imu_grade not in GRADES:
        raise HTTPException(400, f"Unknown imu_grade '{req.imu_grade}'")
    region, trajectory = _build_region_and_trajectory(
        req.waypoints, req.speed_mps, req.turn_rate_dps, req.dt, req.n_grid)
    grade = GRADES[req.imu_grade]
    cfg = ekf.EKFConfig(use_mag=True, use_grav=False)
    sweep = simulate.sweep_1d(region, trajectory, grade, cfg, "mag_noise_std_nt",
                               SWEEP_VALUES, dt=req.dt, n_trials=req.n_trials)
    return {
        "values": SWEEP_VALUES,
        "means": [float(sweep[v].mean()) for v in SWEEP_VALUES],
        "stds": [float(sweep[v].std()) for v in SWEEP_VALUES],
    }


@app.post("/api/sweep/grav")
def sweep_grav(req: SweepRequest):
    if req.imu_grade not in GRADES:
        raise HTTPException(400, f"Unknown imu_grade '{req.imu_grade}'")
    region, trajectory = _build_region_and_trajectory(
        req.waypoints, req.speed_mps, req.turn_rate_dps, req.dt, req.n_grid)
    grade = GRADES[req.imu_grade]
    cfg = ekf.EKFConfig(use_mag=False, use_grav=True)
    sweep = simulate.sweep_1d(region, trajectory, grade, cfg, "grav_noise_std_mgal",
                               SWEEP_VALUES, dt=req.dt, n_trials=req.n_trials)
    return {
        "values": SWEEP_VALUES,
        "means": [float(sweep[v].mean()) for v in SWEEP_VALUES],
        "stds": [float(sweep[v].std()) for v in SWEEP_VALUES],
    }


@app.post("/api/sweep/2d")
def sweep_2d(req: SweepRequest):
    if req.imu_grade not in GRADES:
        raise HTTPException(400, f"Unknown imu_grade '{req.imu_grade}'")
    region, trajectory = _build_region_and_trajectory(
        req.waypoints, req.speed_mps, req.turn_rate_dps, req.dt, req.n_grid)
    grade = GRADES[req.imu_grade]
    # Matched update rates (rather than the 1s/60s simulate default) so the
    # grid shows the actual fusion trade-off instead of being dominated by
    # whichever sensor updates more often -- same reasoning as the CLI sweep.
    cfg = dataclasses.replace(ekf.EKFConfig(), mag_update_interval_s=10.0,
                               grav_update_interval_s=10.0)
    grid = simulate.sweep_2d(region, trajectory, grade, cfg, SWEEP_2D_VALUES, SWEEP_2D_VALUES,
                              dt=req.dt, n_trials=max(2, req.n_trials // 2))
    return {
        "mag_values": SWEEP_2D_VALUES,
        "grav_values": SWEEP_2D_VALUES,
        "grid": grid.tolist(),
    }


@app.post("/api/drift")
def drift(req: DriftRequest):
    trajectory = _waypoints_to_trajectory(
        req.waypoints, req.speed_mps, req.turn_rate_dps, req.dt)

    # Horizons beyond the route's actual flight duration would otherwise
    # silently clamp to the last sample and repeat an identical value
    # (fieldmark/simulate.py:dead_reckoning_drift indexes with
    # min(round(h/dt), n_steps-1)) -- drop those rather than show flat,
    # misleading repeats on a short route.
    duration_s = trajectory.t[-1]
    horizons_s = [h for h in DRIFT_HORIZONS_S if h <= duration_s] or [duration_s]

    by_horizon = simulate.sweep_imu_grade_drift(
        trajectory, list(GRADES.keys()), dt=req.dt,
        horizons_s=horizons_s, n_trials=req.n_trials)
    by_grade = {
        name: [float(by_horizon[name][h].mean()) for h in horizons_s]
        for name in GRADES
    }
    return {"horizons_s": horizons_s, "by_grade": by_grade}
