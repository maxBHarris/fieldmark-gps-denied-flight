"""Magnetic field evaluation using the real IGRF-14 model.

Wraps the `ppigrf` package (pure-Python, validated IGRF-13/14 implementation)
to synthesize the main geomagnetic field at arbitrary latitude/longitude/
altitude/date. IGRF is a low-degree (n<=13) spherical-harmonic model of the
Earth's *core* field -- smooth, long-wavelength (thousands of km). Real
magnetic map-matching ("MagNav") relies on the short-wavelength crustal
anomaly field on top of this, which IGRF does not include (no public,
easily embeddable global crustal model exists at the resolution/size this
project can realistically ship). We add a calibrated synthetic crustal
texture layer on top of the real IGRF core field -- see
`fieldmark.crustal_anomaly` -- so map-matching has the short-wavelength
signal it needs, while the large-scale field is the genuine published model.
"""
import numpy as np
import pandas as pd
import ppigrf

DEFAULT_DATE = pd.Timestamp("2026-01-01")


def total_intensity_grid(lat_grid_deg, lon_grid_deg, alt_km=10.0, date=DEFAULT_DATE):
    """Total field intensity |B| (nT) on a (possibly 2D) lat/lon grid.

    lat_grid_deg, lon_grid_deg: arrays of identical shape (e.g. from
        np.meshgrid), degrees.
    alt_km: height above the WGS84 ellipsoid / IGRF reference, km.
    Returns array of |B| in nT, same shape as inputs.
    """
    lat = np.asarray(lat_grid_deg, dtype=float)
    lon = np.asarray(lon_grid_deg, dtype=float)
    shape = lat.shape
    lat_flat = lat.ravel()
    lon_flat = lon.ravel()
    alt = np.full_like(lat_flat, alt_km)

    # ppigrf.igrf returns (east, north, up) components in nT, geodetic.
    Be, Bn, Bu = ppigrf.igrf(lon_flat, lat_flat, alt, date)
    Be = np.asarray(Be).ravel()
    Bn = np.asarray(Bn).ravel()
    Bu = np.asarray(Bu).ravel()
    Btot = np.sqrt(Be ** 2 + Bn ** 2 + Bu ** 2)
    return Btot.reshape(shape)


def vector_field(lat_deg, lon_deg, alt_km, date=DEFAULT_DATE):
    """Field vector components (east, north, up), nT, at point array(s)."""
    lat = np.atleast_1d(np.asarray(lat_deg, dtype=float))
    lon = np.atleast_1d(np.asarray(lon_deg, dtype=float))
    alt = np.full_like(lat, alt_km) if np.isscalar(alt_km) else np.atleast_1d(alt_km)
    Be, Bn, Bu = ppigrf.igrf(lon, lat, alt, date)
    return np.asarray(Be).ravel(), np.asarray(Bn).ravel(), np.asarray(Bu).ravel()
