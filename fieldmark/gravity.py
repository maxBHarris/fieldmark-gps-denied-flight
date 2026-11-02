"""Gravity disturbance evaluation from the real EGM2008 spherical-harmonic model.

Coefficients come from `data/EGM2008_deg360.gfc`, a degree/order-360
truncation of the published EGM2008 model (Pavlis et al. 2012), downloaded
directly from ICGEM. Degree 360 corresponds to a spatial half-wavelength of
about 55 km at the surface -- band-limiting a high-degree model like this
before synthesis is standard practice in physical geodesy (the omitted
degrees 361-2190 are sub-tens-of-km terrain-correlated signal, well below
what an airborne gravimeter can usefully resolve at normal survey altitude
and speed anyway).

We synthesize the gravity *disturbance* -- the same quantity a map-matching
gravimeter compares against a stored anomaly grid -- via the standard
physical-geodesy spherical harmonic sum:

    T(r, phi, lambda) = (GM/r) * sum_{n=2}^{N} (a/r)^n *
                         sum_{m=0}^{n} [Cnm cos(m lambda) + Snm sin(m lambda)] * Pnm_bar(sin phi)

    delta_g(r, phi, lambda) = -dT/dr
                             = (GM/r^2) * sum_n (n+1) (a/r)^n * sum_m [...] * Pnm_bar(sin phi)

Before synthesis we remove the normal (reference-ellipsoid) even zonal
terms J2/J4/J6 from C20/C40/C60, which is the standard way to turn "the
full gravity field" into "the anomaly relative to the WGS84 reference
ellipsoid" -- without it the result is dominated by the ~1e5 mGal
equatorial-bulge signal instead of the ~10-200 mGal regional texture that
is actually useful for navigation.
"""
import numpy as np

from .constants import DEG2RAD

MGAL_PER_SI = 1.0e5  # 1 mGal = 1e-5 m/s^2

# WGS84 normal-ellipsoid even zonal harmonics (fully normalized Jn), used to
# remove the reference ellipsoid field from the EGM2008 zonal coefficients.
_J2 = 1.08263e-3
_J4 = -2.37091222e-6
_J6 = 6.08347e-9


def _parse_float(tok):
    return float(tok.replace("d", "e").replace("D", "E"))


class GravityModel:
    def __init__(self, gfc_path, nmax=360):
        self.nmax = nmax
        GM, a, C, S = self._load(gfc_path, nmax)
        self.GM = GM
        self.a = a
        self.C = C  # shape (nmax+1, nmax+1), C[n, m]
        self.S = S
        self._remove_normal_field()

    @staticmethod
    def _load(path, nmax):
        GM = None
        a = None
        header_done = False
        C = np.zeros((nmax + 1, nmax + 1))
        S = np.zeros((nmax + 1, nmax + 1))
        with open(path, "r") as f:
            for line in f:
                if not header_done:
                    if line.startswith("earth_gravity_constant"):
                        GM = float(line.split()[1])
                    elif line.startswith("radius"):
                        a = float(line.split()[1])
                    elif line.startswith("end_of_head"):
                        header_done = True
                    continue
                parts = line.split()
                if not parts or parts[0] != "gfc":
                    continue
                n = int(parts[1])
                m = int(parts[2])
                if n > nmax:
                    continue
                C[n, m] = _parse_float(parts[3])
                S[n, m] = _parse_float(parts[4])
        if GM is None or a is None:
            raise ValueError(f"Could not parse header of {path}")
        return GM, a, C, S

    def _remove_normal_field(self):
        # The n=0 (point-mass) and n=1 (origin-at-center-of-mass) terms are
        # common to both the actual and normal potential and cancel out of
        # the *disturbing* potential T = W - U by definition -- so the sum
        # below starts at n=2. Zeroing C[0,0] here lets the synthesis loop
        # run over the full n range without special-casing it.
        self.C[0, 0] = 0.0
        self.C[2, 0] -= -_J2 / np.sqrt(5)
        if self.nmax >= 4:
            self.C[4, 0] -= -_J4 / np.sqrt(9)
        if self.nmax >= 6:
            self.C[6, 0] -= -_J6 / np.sqrt(13)

    def disturbance_mgal(self, lat_deg, lon_deg, alt_km):
        """Gravity disturbance (mGal) at arbitrary-shaped lat/lon/alt arrays."""
        lat = np.asarray(lat_deg, dtype=float)
        lon = np.asarray(lon_deg, dtype=float)
        shape = lat.shape
        lat_flat = lat.ravel()
        lon_flat = lon.ravel()
        alt_m = (np.full_like(lat_flat, alt_km) if np.isscalar(alt_km)
                 else np.asarray(alt_km, dtype=float).ravel()) * 1000.0

        r = self.a + alt_m  # spherical approximation of geocentric radius
        t = np.sin(lat_flat * DEG2RAD)
        u = np.cos(lat_flat * DEG2RAD)
        lam = lon_flat * DEG2RAD
        ratio = self.a / r

        nmax = self.nmax
        npts = lat_flat.size

        # cos(m*lambda), sin(m*lambda) via recursion, kept as list-of-arrays.
        cos_m = [np.ones(npts), np.cos(lam)]
        sin_m = [np.zeros(npts), np.sin(lam)]
        two_cos = 2.0 * cos_m[1]
        for m in range(2, nmax + 1):
            cos_m.append(two_cos * cos_m[-1] - cos_m[-2])
            sin_m.append(two_cos * sin_m[-1] - sin_m[-2])

        total = np.zeros(npts)

        # Running sectorial P_mm, built up as m increases.
        Pmm = np.ones(npts)  # P_00 = 1
        for m in range(0, nmax + 1):
            if m == 0:
                Pmm_cur = np.ones(npts)
            elif m == 1:
                Pmm_cur = np.sqrt(3.0) * u
            else:
                Pmm_cur = u * np.sqrt((2 * m + 1) / (2.0 * m)) * Pmm
            Pmm = Pmm_cur

            Cm = self.C[m:, m]
            Sm = self.S[m:, m]
            if not np.any(Cm) and not np.any(Sm):
                continue

            ratio_pow = ratio ** m
            P_prev2 = None
            P_prev1 = Pmm.copy()

            trig = Cm[0] * cos_m[m] + Sm[0] * sin_m[m]
            total += (m + 1) * ratio_pow * P_prev1 * trig

            for n in range(m + 1, nmax + 1):
                if n == m + 1:
                    Pn = np.sqrt(2 * m + 3) * t * P_prev1
                else:
                    a_nm = np.sqrt(((2 * n - 1) * (2 * n + 1)) / ((n - m) * (n + m)))
                    b_nm = np.sqrt(((2 * n + 1) * (n - m - 1) * (n + m - 1)) /
                                   ((2 * n - 3) * (n - m) * (n + m)))
                    Pn = a_nm * t * P_prev1 - b_nm * P_prev2
                ratio_pow = ratio_pow * ratio
                idx = n - m
                trig = Cm[idx] * cos_m[m] + Sm[idx] * sin_m[m]
                total += (n + 1) * ratio_pow * Pn * trig
                P_prev2 = P_prev1
                P_prev1 = Pn

        dist = (self.GM / r ** 2) * total  # m/s^2
        return (dist * MGAL_PER_SI).reshape(shape)
