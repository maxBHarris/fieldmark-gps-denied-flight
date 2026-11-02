#!/usr/bin/env python
"""Monte Carlo sweep: steady-state position accuracy vs. magnetometer noise,
gravimeter noise, and IMU grade.

Example:
    python scripts/run_noise_sweep.py --n-trials 10 --out-dir results/
"""
import argparse
import dataclasses
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fieldmark.fieldmap import FieldRegion
from fieldmark.trajectory import boustrophedon_survey
from fieldmark.imu import GRADES
from fieldmark import ekf, simulate
from fieldmark.plotting import plot_noise_sweep, plot_sweep_2d, plot_drift_vs_time


def _with(config, **kwargs):
    return dataclasses.replace(config, **kwargs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lat", type=float, default=40.0)
    ap.add_argument("--lon", type=float, default=-105.0)
    ap.add_argument("--half-width", type=float, default=100.0)
    ap.add_argument("--n-grid", type=int, default=121)
    ap.add_argument("--imu-grade", choices=list(GRADES.keys()), default="tactical")
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fr = FieldRegion(args.lat, args.lon, half_width_km=args.half_width, n_grid=args.n_grid)
    tr = boustrophedon_survey(half_width_km=args.half_width * 0.8, dt=args.dt)
    grade = GRADES[args.imu_grade]
    base_cfg = ekf.EKFConfig()

    # Magnetic updates (1/s) are far more frequent than gravity updates
    # (1/60s, reflecting a gravimeter's much slower effective bandwidth), so
    # in a fused run magnetic noise dominates and swamps any visible
    # sensitivity to gravimeter noise. To show each sensor's *own*
    # noise-accuracy relationship, these two sweeps isolate one aiding
    # source at a time; the 2D sweep below shows them fused together.
    print("Sweeping magnetometer noise (magnetic-only aiding)...")
    mag_values = [0.1, 0.3, 1, 3, 10, 30, 100]
    mag_only_cfg = _with(base_cfg, use_mag=True, use_grav=False)
    mag_sweep = simulate.sweep_1d(fr, tr, grade, mag_only_cfg, "mag_noise_std_nt",
                                   mag_values, dt=args.dt, n_trials=args.n_trials)
    fig1 = plot_noise_sweep(mag_sweep, "Magnetometer noise, 1-sigma (nT)",
                             f"Accuracy vs. magnetometer noise, magnetic-only aiding ({grade.name} IMU)")
    fig1.savefig(os.path.join(args.out_dir, "sweep_mag_noise.png"), dpi=150)

    print("Sweeping gravimeter noise (gravity-only aiding)...")
    grav_values = [0.1, 0.3, 1, 3, 10, 30, 100]
    grav_only_cfg = _with(base_cfg, use_mag=False, use_grav=True)
    grav_sweep = simulate.sweep_1d(fr, tr, grade, grav_only_cfg, "grav_noise_std_mgal",
                                    grav_values, dt=args.dt, n_trials=args.n_trials)
    fig2 = plot_noise_sweep(grav_sweep, "Gravimeter noise, 1-sigma (mGal)",
                             f"Accuracy vs. gravimeter noise, gravity-only aiding ({grade.name} IMU)")
    fig2.savefig(os.path.join(args.out_dir, "sweep_grav_noise.png"), dpi=150)

    # Note on IMU grade: once map-matching updates are frequent (the 1 Hz
    # magnetic default here), steady-state accuracy is set mostly by sensor
    # noise and update rate, not IMU grade -- drift between corrections is
    # tiny for *any* grade at 1 Hz. IMU grade instead shows up in how fast
    # position error grows when aiding is unavailable, so that's what this
    # plot shows: pure unaided dead-reckoning drift vs. time, by grade.
    print("Comparing unaided (dead-reckoning) drift by IMU grade...")
    horizons_s = [30, 60, 120, 300, 600, 1200]
    drift = simulate.sweep_imu_grade_drift(tr, list(GRADES.keys()), dt=args.dt,
                                            horizons_s=horizons_s, n_trials=args.n_trials)
    fig3 = plot_drift_vs_time(drift, horizons_s)
    fig3.savefig(os.path.join(args.out_dir, "drift_vs_imu_grade.png"), dpi=150)

    # Matched update rates here (rather than the 1s/60s realistic default)
    # so the plot shows the actual mag/grav fusion trade-off instead of
    # being dominated by whichever sensor happens to update more often.
    print("Running joint 2D sweep (mag x grav, matched update rates)...")
    mag_2d = [0.3, 1, 3, 10, 30]
    grav_2d = [0.3, 1, 3, 10, 30]
    matched_cfg = _with(base_cfg, mag_update_interval_s=10.0, grav_update_interval_s=10.0)
    grid = simulate.sweep_2d(fr, tr, grade, matched_cfg, mag_2d, grav_2d,
                              dt=args.dt, n_trials=max(3, args.n_trials // 2))
    fig4 = plot_sweep_2d(grid, mag_2d, grav_2d)
    fig4.savefig(os.path.join(args.out_dir, "sweep_2d.png"), dpi=150)

    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
