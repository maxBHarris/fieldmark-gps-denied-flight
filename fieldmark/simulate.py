"""Monte Carlo simulation runner: single trials and noise-level sweeps.

A "trial" = one simulated flight: generate noisy IMU measurements for the
ground-truth trajectory, run the EKF (dead reckoning + map-matching
updates), and score the resulting position error. Sweeps repeat this over a
grid of sensor-noise levels / IMU grades and aggregate accuracy statistics,
which is the thing this whole project is meant to show: how position
accuracy depends on magnetometer/gravimeter noise.
"""
from dataclasses import dataclass

import numpy as np

from . import ekf
from .imu import GRADES, simulate_measurements


@dataclass
class TrialSummary:
    tail_rms_m: float      # RMS error over the back half of the flight (steady-state accuracy)
    final_error_m: float
    max_error_m: float
    mag_updates: int
    grav_updates: int


def run_trial(fieldmap, trajectory, grade, config, dt, seed, tail_fraction=0.5):
    """Run one Monte Carlo trial with a fully independent RNG stream."""
    imu_rng = np.random.default_rng(seed)
    ekf_rng = np.random.default_rng(seed + 1_000_000)
    imu_meas = simulate_measurements(trajectory, grade, dt, imu_rng)
    result = ekf.run(trajectory, imu_meas, fieldmap, grade, config, dt, ekf_rng)

    n = len(result.pos_error_m)
    tail = result.pos_error_m[int((1 - tail_fraction) * n):]
    summary = TrialSummary(
        tail_rms_m=float(np.sqrt((tail ** 2).mean())),
        final_error_m=float(result.pos_error_m[-1]),
        max_error_m=float(result.pos_error_m.max()),
        mag_updates=result.mag_updates,
        grav_updates=result.grav_updates,
    )
    return result, summary


def monte_carlo(fieldmap, trajectory, grade, config, dt, n_trials, base_seed=0):
    """Run n_trials independent trials; returns list of TrialSummary."""
    summaries = []
    for i in range(n_trials):
        _, s = run_trial(fieldmap, trajectory, grade, config, dt, base_seed + i)
        summaries.append(s)
    return summaries


def _tail_rms_array(summaries):
    return np.array([s.tail_rms_m for s in summaries])


def sweep_1d(fieldmap, trajectory, grade, base_config, param_name, values,
             dt, n_trials=10, base_seed=0):
    """Sweep one EKFConfig field (e.g. 'mag_noise_std_nt') over `values`.

    Returns dict: value -> array of per-trial tail-RMS errors (n_trials,).
    """
    results = {}
    for v in values:
        cfg = _replace(base_config, **{param_name: v})
        summaries = monte_carlo(fieldmap, trajectory, grade, cfg, dt, n_trials, base_seed)
        results[v] = _tail_rms_array(summaries)
    return results


def sweep_2d(fieldmap, trajectory, grade, base_config, mag_values, grav_values,
             dt, n_trials=6, base_seed=0):
    """Sweep magnetometer noise x gravimeter noise jointly.

    Returns a 2D array of shape (len(grav_values), len(mag_values)) holding
    the median tail-RMS error (m) for each (grav_noise, mag_noise) pair.
    """
    grid = np.zeros((len(grav_values), len(mag_values)))
    for i, gv in enumerate(grav_values):
        for j, mv in enumerate(mag_values):
            cfg = _replace(base_config, mag_noise_std_nt=mv, grav_noise_std_mgal=gv)
            summaries = monte_carlo(fieldmap, trajectory, grade, cfg, dt, n_trials, base_seed)
            grid[i, j] = np.median(_tail_rms_array(summaries))
    return grid


def dead_reckoning_drift(trajectory, grade, dt, horizons_s, n_trials=8, base_seed=0):
    """Pure unaided (no map updates) dead-reckoning drift, by IMU grade.

    This isolates the IMU's own error growth -- no EKF map-matching, and
    zero initial position/velocity/heading uncertainty -- so differences
    across IMU grades come *only* from their bias/noise specs, not from
    Kalman-gain tuning effects (which do matter once map-matching is
    active, see the mag/gravity noise sweeps, but would otherwise confound
    a direct "how good is this IMU" comparison). This is the standard
    "free-inertial drift" comparison used in INS datasheets.

    Returns dict: horizon_s -> array of per-trial position errors (m).
    """
    cfg = ekf.EKFConfig(use_mag=False, use_grav=False, init_pos_std_m=0.0,
                         init_speed_std_mps=0.0, init_heading_std_rad=0.0)
    n_steps = len(trajectory.t)
    results = {h: [] for h in horizons_s}
    for i in range(n_trials):
        seed = base_seed + i
        imu_rng = np.random.default_rng(seed)
        ekf_rng = np.random.default_rng(seed + 1_000_000)
        imu_meas = simulate_measurements(trajectory, grade, dt, imu_rng)
        # A dummy FieldRegion-like object is not needed: with use_mag/use_grav
        # False, ekf.run never calls fieldmap.* methods, so None is safe here.
        result = ekf.run(trajectory, imu_meas, None, grade, cfg, dt, ekf_rng)
        for h in horizons_s:
            idx = min(int(round(h / dt)), n_steps - 1)
            results[h].append(result.pos_error_m[idx])
    return {h: np.array(v) for h, v in results.items()}


def sweep_imu_grade_drift(trajectory, grade_names, dt, horizons_s, n_trials=8, base_seed=0):
    """dead_reckoning_drift() for each named IMU grade.

    Returns dict: grade_name -> {horizon_s -> array of errors (m)}.
    """
    return {name: dead_reckoning_drift(trajectory, GRADES[name], dt, horizons_s,
                                        n_trials, base_seed)
            for name in grade_names}


def _replace(config, **kwargs):
    from dataclasses import replace
    return replace(config, **kwargs)
