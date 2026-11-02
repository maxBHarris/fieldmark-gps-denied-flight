"""Extended Kalman Filter fusing dead-reckoned INS with magnetic/gravity
map-matching measurements.

State vector x = [e, n, v, heading, b_gyro, b_accel]:
    e, n       -- position in the local tangent plane, m
    v          -- along-track speed, m/s
    heading    -- rad, 0 = north, positive clockwise
    b_gyro     -- gyro bias, rad/s
    b_accel    -- forward accelerometer bias, m/s^2

Process model: a coordinated-turn dead-reckoning mechanization driven by the
(noisy, biased) IMU measurements -- exactly what an inertial nav system does
between fixes. Measurement model: the magnetic total-intensity and/or
gravity-disturbance value interpolated from the onboard `FieldRegion` map at
the current position estimate, compared against the (noisy) sensor reading
at the vehicle's true position. The Jacobian of each measurement is the
map's analytic spatial gradient at the current estimate -- the position
information in map-matching navigation comes entirely from that gradient:
a flat part of the field gives zero information, a steep gradient gives a
strong fix, which is exactly what this filter will show.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class EKFConfig:
    init_pos_std_m: float = 50.0
    init_speed_std_mps: float = 1.0
    init_heading_std_rad: float = np.deg2rad(1.0)
    mag_noise_std_nt: float = 1.0
    grav_noise_std_mgal: float = 1.0
    mag_update_interval_s: float = 1.0
    grav_update_interval_s: float = 60.0
    use_mag: bool = True
    use_grav: bool = True
    bias_rw_fraction: float = 0.01  # fraction of bias std used as per-sqrt(s) random-walk rate

    # Process-noise floor: a filter tuned to trust an IMU's advertised spec
    # exactly becomes arbitrarily overconfident as that spec improves (P
    # shrinks, Kalman gain shrinks, and the filter stops listening to new
    # measurements) -- and any real system has *some* unmodeled error
    # (linearization, timing, scale factor) that a zero-floor filter can't
    # absorb, so it can latch onto a bad estimate and never recover. These
    # floors set a minimum effective noise level used only for the filter's
    # own Q/P bookkeeping (the simulated "true" IMU errors are unaffected),
    # which is standard practice for robust Kalman filter tuning.
    min_gyro_arw_dpsqrth: float = 0.05
    min_accel_vrw_ugsqrthz: float = 10.0
    min_gyro_bias_dph: float = 1.0
    min_accel_bias_ug: float = 50.0

    # Innovation (outlier) gate: reject a map-matching update if the
    # measurement residual is more than this many predicted standard
    # deviations away from zero. This is standard practice in real
    # map-matching systems -- if the position estimate has drifted far
    # enough that the local field gradient is no longer a valid linear
    # approximation (or the vehicle is simply in a textured/ambiguous part
    # of the map), accepting that update can do more harm than good; better
    # to coast on the INS until a trustworthy fix is available.
    innovation_gate_sigma: float = 5.0


@dataclass
class EKFResult:
    t: np.ndarray
    est_e: np.ndarray
    est_n: np.ndarray
    est_heading: np.ndarray   # rad, EKF's estimated heading (for drawing a corridor
                              # perpendicular to the track -- see fieldmark.simulate)
    true_e: np.ndarray
    true_n: np.ndarray
    pos_error_m: np.ndarray
    pos_std_m: np.ndarray     # sqrt of trace(P_pos), 1-sigma
    mag_updates: int = 0
    grav_updates: int = 0


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def run(trajectory, imu_meas, fieldmap, grade, config, dt, rng):
    """Run one EKF trial. `trajectory` is ground truth; `imu_meas` are the
    noisy IMU measurements already generated for it (see fieldmark.imu)."""
    n_steps = len(trajectory.t)

    x = np.array([
        trajectory.east[0] + rng.normal(0, config.init_pos_std_m),
        trajectory.north[0] + rng.normal(0, config.init_pos_std_m),
        trajectory.speed + rng.normal(0, config.init_speed_std_mps),
        trajectory.heading[0] + rng.normal(0, config.init_heading_std_rad),
        0.0,
        0.0,
    ])
    from .imu import IMUGrade, gyro_bias_rad_s, accel_bias_mps2, gyro_noise_std_rad_s, accel_noise_std_mps2
    # Filter-internal tuning uses the IMU grade clamped to the process-noise
    # floors above; the *true* simulated IMU errors (in `imu_meas`, from the
    # real `grade`) are unaffected.
    filter_grade = IMUGrade(
        name=grade.name,
        gyro_bias_dph=max(grade.gyro_bias_dph, config.min_gyro_bias_dph),
        gyro_arw_dphsqrth=max(grade.gyro_arw_dphsqrth, config.min_gyro_arw_dpsqrth),
        accel_bias_ug=max(grade.accel_bias_ug, config.min_accel_bias_ug),
        accel_vrw_ugsqrthz=max(grade.accel_vrw_ugsqrthz, config.min_accel_vrw_ugsqrthz),
    )
    gyro_bias_std = gyro_bias_rad_s(filter_grade)
    accel_bias_std = accel_bias_mps2(filter_grade)

    P = np.diag([
        config.init_pos_std_m ** 2,
        config.init_pos_std_m ** 2,
        config.init_speed_std_mps ** 2,
        config.init_heading_std_rad ** 2,
        gyro_bias_std ** 2,
        accel_bias_std ** 2,
    ])

    gyro_noise_var = gyro_noise_std_rad_s(filter_grade, dt) ** 2
    accel_noise_var = accel_noise_std_mps2(filter_grade, dt) ** 2
    q_bgyro = (config.bias_rw_fraction * gyro_bias_std) ** 2 * dt
    q_baccel = (config.bias_rw_fraction * accel_bias_std) ** 2 * dt

    est_e = np.empty(n_steps)
    est_n = np.empty(n_steps)
    est_heading = np.empty(n_steps)
    pos_std = np.empty(n_steps)

    mag_updates = 0
    grav_updates = 0
    mag_every = max(1, int(round(config.mag_update_interval_s / dt)))
    grav_every = max(1, int(round(config.grav_update_interval_s / dt)))

    for k in range(n_steps):
        est_e[k] = x[0]
        est_n[k] = x[1]
        est_heading[k] = x[3]
        pos_std[k] = np.sqrt(P[0, 0] + P[1, 1])

        # -- measurement updates (map matching) --
        true_e_k, true_n_k = trajectory.east[k], trajectory.north[k]

        if config.use_mag and k % mag_every == 0 and fieldmap.in_bounds(x[0], x[1]):
            z = fieldmap.mag_value(true_e_k, true_n_k) + rng.normal(0, config.mag_noise_std_nt)
            h = fieldmap.mag_value(x[0], x[1])
            dhde, dhdn = fieldmap.mag_grad(x[0], x[1])
            H = np.array([dhde, dhdn, 0.0, 0.0, 0.0, 0.0])
            R = config.mag_noise_std_nt ** 2
            x, P, accepted = _gated_scalar_update(x, P, z - h, H, R, config.innovation_gate_sigma)
            mag_updates += int(accepted)

        if config.use_grav and k % grav_every == 0 and fieldmap.in_bounds(x[0], x[1]):
            z = fieldmap.grav_value(true_e_k, true_n_k) + rng.normal(0, config.grav_noise_std_mgal)
            h = fieldmap.grav_value(x[0], x[1])
            dhde, dhdn = fieldmap.grav_grad(x[0], x[1])
            H = np.array([dhde, dhdn, 0.0, 0.0, 0.0, 0.0])
            R = config.grav_noise_std_mgal ** 2
            x, P, accepted = _gated_scalar_update(x, P, z - h, H, R, config.innovation_gate_sigma)
            grav_updates += int(accepted)

        if k == n_steps - 1:
            break

        # -- propagate (INS mechanization) --
        accel_m = imu_meas.accel_fwd[k]
        gyro_m = imu_meas.gyro[k]

        e, n, v, hdg, b_gyro, b_acc = x
        accel_true = accel_m - b_acc
        gyro_true = gyro_m - b_gyro

        e_next = e + v * np.sin(hdg) * dt
        n_next = n + v * np.cos(hdg) * dt
        v_next = v + accel_true * dt
        hdg_next = _wrap(hdg + gyro_true * dt)

        F = np.eye(6)
        F[0, 2] = np.sin(hdg) * dt
        F[0, 3] = v * np.cos(hdg) * dt
        F[1, 2] = np.cos(hdg) * dt
        F[1, 3] = -v * np.sin(hdg) * dt
        F[2, 5] = -dt
        F[3, 4] = -dt

        G = np.zeros((6, 2))  # maps [accel_noise, gyro_noise] into state noise
        G[2, 0] = dt
        G[3, 1] = dt
        Qinput = np.diag([accel_noise_var, gyro_noise_var])

        Q = G @ Qinput @ G.T
        Q[4, 4] += q_bgyro
        Q[5, 5] += q_baccel

        x = np.array([e_next, n_next, v_next, hdg_next, b_gyro, b_acc])
        P = F @ P @ F.T + Q

    pos_error = np.sqrt((est_e - trajectory.east) ** 2 + (est_n - trajectory.north) ** 2)

    return EKFResult(
        t=trajectory.t, est_e=est_e, est_n=est_n, est_heading=est_heading,
        true_e=trajectory.east, true_n=trajectory.north,
        pos_error_m=pos_error, pos_std_m=pos_std,
        mag_updates=mag_updates, grav_updates=grav_updates,
    )


def _gated_scalar_update(x, P, innovation, H, R, gate_sigma):
    """Soft outlier gate: if the innovation exceeds the gate, inflate R
    (rather than hard-rejecting) so the update still nudges the estimate
    but can't dominate it. A hard reject would leave P unchanged on a
    rejected update, which -- if the estimate is genuinely drifting --
    stops it from ever shrinking again and permanently locks the filter
    out of the one thing that could fix the drift.
    """
    S_nominal = H @ P @ H.T + R
    ratio = abs(innovation) / (gate_sigma * np.sqrt(S_nominal))
    accepted = ratio <= 1.0
    R_eff = R if accepted else R * ratio ** 2
    S = H @ P @ H.T + R_eff
    x_new, P_new = _scalar_update(x, P, innovation, H, R_eff, S=S)
    return x_new, P_new, accepted


def _scalar_update(x, P, innovation, H, R, S=None):
    if S is None:
        S = H @ P @ H.T + R
    K = (P @ H) / S
    x_new = x + K * innovation
    x_new[3] = _wrap(x_new[3])
    I = np.eye(len(x))
    P_new = (I - np.outer(K, H)) @ P
    return x_new, P_new
