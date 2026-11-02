"""Build and cache magnetic/gravity reference maps for a flight region.

A `FieldRegion` synthesizes the magnetic total-intensity field (real IGRF-14
core field + synthetic crustal texture, see `fieldmark.crustal_anomaly`) and
the real EGM2008-derived gravity disturbance field once, on a regular grid
over a local tangent-plane area, then wraps each grid in a bicubic spline
(`scipy.interpolate.RectBivariateSpline`). The spline gives fast O(1)
lookups *and* exact analytic partial derivatives, which the EKF uses
directly as its measurement Jacobian -- this mirrors how a real system
would carry a precomputed onboard map rather than re-evaluating spherical
harmonics in the navigation loop.
"""
import hashlib
import os

import numpy as np
from scipy.interpolate import RectBivariateSpline

from .constants import LocalTangentPlane
from . import igrf
from . import crustal_anomaly
from .gravity import GravityModel

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_CACHE_DIR = os.path.join(_DATA_DIR, "cache")
_GFC_PATH = os.path.join(_DATA_DIR, "EGM2008_deg360.gfc")

_gravity_model = None


def _get_gravity_model():
    global _gravity_model
    if _gravity_model is None:
        _gravity_model = GravityModel(_GFC_PATH, nmax=360)
    return _gravity_model


class FieldRegion:
    """Magnetic + gravity reference maps over a square region.

    lat0_deg, lon0_deg: region center.
    half_width_km: region extends +/- this distance (local east/north) from center.
    n_grid: number of grid points per axis.
    alt_km: nominal flight altitude used to evaluate the fields (both fields
        vary slowly with altitude over typical cruise-altitude ranges, so a
        single nominal altitude is used for the stored map, matching how
        real onboard anomaly maps are referenced to a nominal survey
        altitude).
    mag_rms_nt / mag_corr_km: crustal anomaly texture parameters.
    """

    def __init__(self, lat0_deg, lon0_deg, half_width_km=100.0, n_grid=161,
                 alt_km=8.0, mag_rms_nt=120.0, mag_corr_km=15.0, use_cache=True):
        self.lat0 = lat0_deg
        self.lon0 = lon0_deg
        self.half_width_km = half_width_km
        self.n_grid = n_grid
        self.alt_km = alt_km
        self.mag_rms_nt = mag_rms_nt
        self.mag_corr_km = mag_corr_km
        self.ltp = LocalTangentPlane(lat0_deg, lon0_deg)

        half_m = half_width_km * 1000.0
        self.east = np.linspace(-half_m, half_m, n_grid)
        self.north = np.linspace(-half_m, half_m, n_grid)

        cache_path = self._cache_path()
        if use_cache and os.path.exists(cache_path):
            data = np.load(cache_path)
            mag_grid = data["mag_grid"]
            grav_grid = data["grav_grid"]
        else:
            mag_grid, grav_grid = self._synthesize()
            if use_cache:
                os.makedirs(_CACHE_DIR, exist_ok=True)
                np.savez(cache_path, mag_grid=mag_grid, grav_grid=grav_grid)

        self.mag_grid = mag_grid
        self.grav_grid = grav_grid
        self._mag_spline = RectBivariateSpline(self.north, self.east, mag_grid)
        self._grav_spline = RectBivariateSpline(self.north, self.east, grav_grid)

    def _cache_path(self):
        key = f"{self.lat0:.4f}_{self.lon0:.4f}_{self.half_width_km}_{self.n_grid}_{self.alt_km}_{self.mag_rms_nt}_{self.mag_corr_km}"
        h = hashlib.sha1(key.encode()).hexdigest()[:16]
        return os.path.join(_CACHE_DIR, f"fieldmap_{h}.npz")

    def _synthesize(self):
        EE, NN = np.meshgrid(self.east, self.north)
        lat_grid, lon_grid = self.ltp.en_to_ll(EE, NN)

        core = igrf.total_intensity_grid(lat_grid, lon_grid, alt_km=self.alt_km)
        seed = crustal_anomaly.region_seed(self.lat0, self.lon0)
        texture = crustal_anomaly.synthesize(
            self.east, self.north, seed,
            rms_nt=self.mag_rms_nt, corr_length_m=self.mag_corr_km * 1000.0,
        )
        mag_grid = core + texture

        gm = _get_gravity_model()
        grav_grid = gm.disturbance_mgal(lat_grid, lon_grid, self.alt_km)

        return mag_grid, grav_grid

    # -- lookups (local ENU meters in, field value / partials out) --------
    def mag_value(self, east_m, north_m):
        return self._mag_spline.ev(north_m, east_m)

    def mag_grad(self, east_m, north_m):
        """Returns (d(mag)/d(east), d(mag)/d(north)) in nT/m."""
        d_e = self._mag_spline.ev(north_m, east_m, dx=0, dy=1)
        d_n = self._mag_spline.ev(north_m, east_m, dx=1, dy=0)
        return d_e, d_n

    def grav_value(self, east_m, north_m):
        return self._grav_spline.ev(north_m, east_m)

    def grav_grad(self, east_m, north_m):
        """Returns (d(grav)/d(east), d(grav)/d(north)) in mGal/m."""
        d_e = self._grav_spline.ev(north_m, east_m, dx=0, dy=1)
        d_n = self._grav_spline.ev(north_m, east_m, dx=1, dy=0)
        return d_e, d_n

    def in_bounds(self, east_m, north_m, margin_m=0.0):
        half = self.half_width_km * 1000.0 - margin_m
        return (np.abs(east_m) <= half) & (np.abs(north_m) <= half)
