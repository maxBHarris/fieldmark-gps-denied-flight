"""Generate ground-truth flight trajectories in the local tangent plane.

Uses a simple coordinated-turn kinematic (unicycle) model: constant true
airspeed, with heading changing at a constant turn rate during turns. This
is the standard simplification used for regional-scale aided-INS studies --
it keeps the IMU mechanization and EKF process model in flat 2D Cartesian
coordinates, while the field maps underneath are still evaluated from real
global models.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Trajectory:
    t: np.ndarray          # s
    east: np.ndarray       # m
    north: np.ndarray      # m
    heading: np.ndarray    # rad, 0 = north, positive clockwise (toward east)
    speed: float           # m/s, constant true airspeed
    turnrate: np.ndarray   # rad/s, signed


def _leg_then_turn(start_e, start_n, heading, leg_len_m, turn_angle_rad,
                    speed, turn_rate, dt):
    """One straight leg followed by a constant-rate turn. Returns samples."""
    samples = []
    # straight leg
    n_leg = max(1, int(round(leg_len_m / speed / dt)))
    e, n = start_e, start_n
    for _ in range(n_leg):
        samples.append((e, n, heading, 0.0))
        e += speed * np.sin(heading) * dt
        n += speed * np.cos(heading) * dt
    # turn
    if abs(turn_angle_rad) > 1e-9:
        signed_rate = turn_rate if turn_angle_rad > 0 else -turn_rate
        n_turn = max(1, int(round(abs(turn_angle_rad) / turn_rate / dt)))
        for _ in range(n_turn):
            samples.append((e, n, heading, signed_rate))
            heading += signed_rate * dt
            e += speed * np.sin(heading) * dt
            n += speed * np.cos(heading) * dt
    return samples, e, n, heading


def boustrophedon_survey(half_width_km=80.0, leg_spacing_km=20.0,
                          speed_mps=180.0, dt=1.0, n_legs=None):
    """A back-and-forth parallel-leg survey pattern (common airborne survey
    flight profile) that stays within +/- half_width_km of the region center.

    A coordinated 180-degree turn at constant speed and turn rate traces a
    semicircle whose diameter is 2*v/omega -- so the turn rate is chosen
    such that that diameter equals leg_spacing_km, which lines the next leg
    up correctly without any extra lateral offset logic.
    """
    half_m = half_width_km * 1000.0
    leg_len_m = 2 * half_m * 0.9
    spacing_m = leg_spacing_km * 1000.0
    turn_rate = 2 * speed_mps / spacing_m  # rad/s, so turn diameter = spacing_m

    if n_legs is None:
        n_legs = max(2, int((2 * half_m * 0.9) / spacing_m))

    e0 = -half_m * 0.8
    n0 = -half_m * 0.9
    heading = 0.0  # start heading north

    all_samples = []
    e, n = e0, n0
    for i in range(n_legs):
        turn_angle = np.pi if i < n_legs - 1 else 0.0
        if i % 2 == 1:
            turn_angle = -turn_angle
        samples, e, n, heading = _leg_then_turn(
            e, n, heading, leg_len_m, turn_angle, speed_mps, turn_rate, dt)
        all_samples.extend(samples)

    east = np.array([s[0] for s in all_samples])
    north = np.array([s[1] for s in all_samples])
    hdg = np.array([s[2] for s in all_samples])
    turnrate = np.array([s[3] for s in all_samples])
    t = np.arange(len(all_samples)) * dt

    return Trajectory(t=t, east=east, north=north, heading=hdg,
                       speed=speed_mps, turnrate=turnrate)


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _turn_then_leg(start_e, start_n, heading, target_e, target_n,
                    speed, turn_rate, dt):
    """Turn to face (target_e, target_n), then fly straight to it.

    Turning first (rather than turning after the straight leg, like
    `_leg_then_turn`) means each leg re-aims at wherever the vehicle
    actually ended up -- including any corner-cutting drift from the
    previous turn -- so course corrections don't compound over a chain of
    waypoints.

    Two turn passes: a turn is a curved arc, not a rotation in place, so
    the bearing to target has generally shifted a bit by the time the
    first pass (aimed from the pre-turn position) finishes -- a second,
    usually much shorter, pass re-aims from the actual post-turn position.
    Both passes record real turnrate samples. An earlier version of this
    function instead silently reassigned `heading` to the recomputed
    bearing without a matching turn -- which fed the IMU-derived gyro
    "truth" signal (built from the recorded turnrate, see
    fieldmark/imu.py:simulate_measurements) a heading jump it never
    actually measured, so the EKF (which only ever sees gyro
    measurements, never ground-truth heading) silently diverged off that
    invisible bias for the rest of the leg -- confirmed via EKF trial logs
    showing accuracy collapsing immediately after corner-heavy legs even
    though both true and estimated positions stayed within the field map.
    """
    samples = []
    e, n = start_e, start_n
    for _ in range(2):
        target_heading = np.arctan2(target_e - e, target_n - n)
        turn_angle = _wrap(target_heading - heading)
        if abs(turn_angle) <= 1e-9:
            continue
        signed_rate = turn_rate if turn_angle > 0 else -turn_rate
        n_turn = max(1, int(round(abs(turn_angle) / turn_rate / dt)))
        for _ in range(n_turn):
            samples.append((e, n, heading, signed_rate))
            heading += signed_rate * dt
            e += speed * np.sin(heading) * dt
            n += speed * np.cos(heading) * dt
        heading = target_heading
    leg_len_m = np.hypot(target_e - e, target_n - n)
    n_leg = max(1, int(round(leg_len_m / speed / dt)))
    for _ in range(n_leg):
        samples.append((e, n, heading, 0.0))
        e += speed * np.sin(heading) * dt
        n += speed * np.cos(heading) * dt
    return samples, e, n, heading


def from_waypoints(waypoints_en, speed_mps=180.0, turn_rate_dps=3.0, dt=1.0):
    """Build a Trajectory flying through user-specified waypoints -- straight
    legs connected by coordinated turns, the shape used for paths planned
    interactively (e.g. clicked on a map), unlike the fixed patterns above.

    waypoints_en: sequence of (east_m, north_m) pairs, at least 2.
    """
    if len(waypoints_en) < 2:
        raise ValueError("from_waypoints needs at least 2 waypoints")

    turn_rate = np.deg2rad(turn_rate_dps)
    n_legs = len(waypoints_en) - 1

    e, n = waypoints_en[0]
    e1, n1 = waypoints_en[1]
    heading = np.arctan2(e1 - e, n1 - n)  # start already aimed at the first waypoint

    all_samples = []
    for i in range(n_legs):
        e1, n1 = waypoints_en[i + 1]
        samples, e, n, heading = _turn_then_leg(
            e, n, heading, e1, n1, speed_mps, turn_rate, dt)
        all_samples.extend(samples)

    east = np.array([s[0] for s in all_samples])
    north = np.array([s[1] for s in all_samples])
    hdg = np.array([s[2] for s in all_samples])
    turnrate = np.array([s[3] for s in all_samples])
    t = np.arange(len(all_samples)) * dt

    return Trajectory(t=t, east=east, north=north, heading=hdg,
                       speed=speed_mps, turnrate=turnrate)


def straight_and_level(length_km=150.0, speed_mps=180.0, dt=1.0,
                        heading_deg=35.0, start_offset_km=(-60.0, -60.0)):
    """A single straight leg -- the simplest useful case, and the hardest
    for map matching (no turns means the gravity/magnetic gradient is only
    observed along one direction, so cross-track error is poorly observed).
    """
    heading = np.deg2rad(heading_deg)
    n_pts = max(2, int(length_km * 1000.0 / speed_mps / dt))
    t = np.arange(n_pts) * dt
    e0, n0 = start_offset_km[0] * 1000.0, start_offset_km[1] * 1000.0
    east = e0 + speed_mps * np.sin(heading) * t
    north = n0 + speed_mps * np.cos(heading) * t
    hdg = np.full(n_pts, heading)
    turnrate = np.zeros(n_pts)
    return Trajectory(t=t, east=east, north=north, heading=hdg,
                       speed=speed_mps, turnrate=turnrate)
