#!/usr/bin/env python
"""Build (or load from cache) the magnetic + gravity reference maps for a
region and plot them.

Example:
    python scripts/build_fieldmaps.py --lat 40.0 --lon -105.0 --half-width 100
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt

from fieldmark.fieldmap import FieldRegion
from fieldmark.plotting import plot_field_maps


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lat", type=float, default=40.0, help="Region center latitude (deg)")
    ap.add_argument("--lon", type=float, default=-105.0, help="Region center longitude (deg)")
    ap.add_argument("--half-width", type=float, default=100.0, help="Region half-width (km)")
    ap.add_argument("--n-grid", type=int, default=121, help="Grid points per axis")
    ap.add_argument("--alt-km", type=float, default=8.0, help="Nominal flight altitude (km)")
    ap.add_argument("--out", default=None, help="Save plot to this path instead of showing")
    args = ap.parse_args()

    print(f"Building field maps for ({args.lat}, {args.lon}), "
          f"+/-{args.half_width} km, {args.n_grid}x{args.n_grid} grid ...")
    fr = FieldRegion(args.lat, args.lon, half_width_km=args.half_width,
                      n_grid=args.n_grid, alt_km=args.alt_km)
    print(f"Magnetic field range: {fr.mag_grid.min():.1f} to {fr.mag_grid.max():.1f} nT")
    print(f"Gravity disturbance range: {fr.grav_grid.min():.1f} to {fr.grav_grid.max():.1f} mGal")

    fig = plot_field_maps(fr)
    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Saved plot to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
