"""Shared physical constants and coordinate conversions."""
import numpy as np

# WGS84 ellipsoid
WGS84_A = 6378137.0          # semi-major axis, m
WGS84_F = 1 / 298.257223563  # flattening
WGS84_B = WGS84_A * (1 - WGS84_F)
WGS84_E2 = 1 - (WGS84_B / WGS84_A) ** 2

EARTH_MEAN_RADIUS = 6371008.7714  # m, IUGG mean radius
G0 = 9.80665  # standard gravity, m/s^2
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi


def meridional_radius(lat_deg):
    """Radius of curvature in the meridian (N-S), WGS84, meters."""
    lat = np.asarray(lat_deg) * DEG2RAD
    s = np.sin(lat)
    return WGS84_A * (1 - WGS84_E2) / (1 - WGS84_E2 * s ** 2) ** 1.5


def prime_vertical_radius(lat_deg):
    """Radius of curvature in the prime vertical (E-W), WGS84, meters."""
    lat = np.asarray(lat_deg) * DEG2RAD
    s = np.sin(lat)
    return WGS84_A / np.sqrt(1 - WGS84_E2 * s ** 2)


class LocalTangentPlane:
    """Flat-Earth local ENU (east, north) tangent plane about a reference lat/lon.

    Valid for regional-scale flights (up to a few hundred km) -- the kind of
    area a single map-matching sortie would cover. Field maps themselves are
    still evaluated on the ellipsoid/sphere using true spherical-harmonic
    models; this class only linearizes the *vehicle kinematics* so the IMU
    mechanization and EKF can use simple Cartesian equations of motion.
    """

    def __init__(self, lat0_deg, lon0_deg):
        self.lat0 = lat0_deg
        self.lon0 = lon0_deg
        self.R_n = meridional_radius(lat0_deg)
        self.R_e = prime_vertical_radius(lat0_deg)

    def ll_to_en(self, lat_deg, lon_deg):
        """Convert lat/lon (deg) to local (east, north) meters."""
        lat_deg = np.asarray(lat_deg)
        lon_deg = np.asarray(lon_deg)
        north = (lat_deg - self.lat0) * DEG2RAD * self.R_n
        east = (lon_deg - self.lon0) * DEG2RAD * self.R_e * np.cos(self.lat0 * DEG2RAD)
        return east, north

    def en_to_ll(self, east_m, north_m):
        """Convert local (east, north) meters to lat/lon (deg)."""
        east_m = np.asarray(east_m)
        north_m = np.asarray(north_m)
        lat_deg = self.lat0 + (north_m / self.R_n) * RAD2DEG
        lon_deg = self.lon0 + (east_m / (self.R_e * np.cos(self.lat0 * DEG2RAD))) * RAD2DEG
        return lat_deg, lon_deg
