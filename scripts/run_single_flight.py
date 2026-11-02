#!/usr/bin/env python
"""Simulate one GPS-denied flight: INS + magnetic/gravity map-matching EKF.

Example:
    python scripts/run_single_flight.py --imu-grade tactical --mag-noise 1 --grav-noise 2
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt

from fieldmark.fieldmap import FieldRegion
from fieldmark.trajectory import boustrophedon_survey, straight_and_level
from fieldmark.imu import GRADES
from fieldmark import ekf, simulate
from fieldmark.plotting import plot_trajectory_error


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lat", type=float, default=40.0)
    ap.add_argument("--lon", type=float, default=-105.0)
    ap.add_argument("--half-width", type=float, default=100.0)
    ap.add_argument("--n-grid", type=int, default=121)
    ap.add_argument("--trajectory", choices=["survey", "straight"], default="survey")
    ap.add_argument("--imu-grade", choices=list(GRADES.keys()), default="tactical")
    ap.add_argument("--mag-noise", type=float, default=1.0, help="Magnetometer noise, 1-sigma nT")
    ap.add_argument("--grav-noise", type=float, default=2.0, help="Gravimeter noise, 1-sigma mGal")
    ap.add_argument("--no-mag", action="store_true", help="Disable magnetic updates")
    ap.add_argument("--no-grav", action="store_true", help="Disable gravity updates")
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fr = FieldRegion(args.lat, args.lon, half_width_km=args.half_width, n_grid=args.n_grid)

    if args.trajectory == "survey":
        tr = boustrophedon_survey(half_width_km=args.half_width * 0.8, dt=args.dt)
    else:
        tr = straight_and_level(length_km=args.half_width * 1.8, dt=args.dt)

    grade = GRADES[args.imu_grade]
    cfg = ekf.EKFConfig(
        mag_noise_std_nt=args.mag_noise,
        grav_noise_std_mgal=args.grav_noise,
        use_mag=not args.no_mag,
        use_grav=not args.no_grav,
    )

    result, summary = simulate.run_trial(fr, tr, grade, cfg, dt=args.dt, seed=args.seed)

    print(f"IMU grade: {grade.name}")
    print(f"Mag noise: {args.mag_noise} nT (enabled={cfg.use_mag}), "
          f"Grav noise: {args.grav_noise} mGal (enabled={cfg.use_grav})")
    print(f"Map updates: {result.mag_updates} magnetic, {result.grav_updates} gravity")
    print(f"Final position error: {summary.final_error_m:.1f} m")
    print(f"Max position error:   {summary.max_error_m:.1f} m")
    print(f"Steady-state RMS error (back half of flight): {summary.tail_rms_m:.1f} m")

    fig = plot_trajectory_error(result, fieldmap=fr)
    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Saved plot to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
