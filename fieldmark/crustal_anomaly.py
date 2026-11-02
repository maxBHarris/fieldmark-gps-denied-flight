"""Synthetic crustal magnetic anomaly texture.

IGRF (`fieldmark.igrf`) only models the Earth's smooth core field
(degree <= 13, wavelengths of thousands of km) -- essentially flat over the
scale of a single flight leg, so it carries almost no map-matching
information by itself. Real airborne magnetic navigation ("MagNav") keys off
the short-wavelength *crustal* anomaly field (local rock magnetization),
which requires a high-degree model (e.g. EMM2017, degree 790) or a compiled
survey grid (e.g. NOAA EMAG2) -- too large to embed or reliably fetch here.

Instead we add a statistically-calibrated synthetic anomaly layer on top of
the real IGRF field: a 2D fractal (power-law spectrum) surface, FFT-
synthesized directly on the local tangent plane, with an RMS amplitude and
dominant correlation length chosen to match published crustal-anomaly
literature (typically tens to a few hundred nT amplitude, with a spectral
falloff of roughly k^-3 over useful wavelengths of a few to a few tens of
km -- see e.g. Maus 2008, "World Digital Magnetic Anomaly Map"). This layer
is clearly a model, not a measurement, and is deterministic given a seed so
a region's map is stable across calls.
"""
import numpy as np


def synthesize(east_m, north_m, seed, rms_nt=120.0, beta=3.0,
               corr_length_m=15000.0):
    """FFT-synthesize a fractal anomaly surface on a regular (east, north) grid.

    east_m, north_m: 1D arrays of grid coordinates (regular spacing), meters.
    seed: int, deterministic RNG seed for this region.
    rms_nt: target RMS amplitude of the anomaly field, nT.
    beta: power-law spectral exponent (S(k) ~ k^-beta).
    corr_length_m: wavelength beyond which the spectrum is flattened
        (controls the dominant feature size), meters.

    Returns a 2D array, shape (len(north_m), len(east_m)), nT.
    """
    ny, nx = len(north_m), len(east_m)
    dx = east_m[1] - east_m[0] if nx > 1 else 1000.0
    dy = north_m[1] - north_m[0] if ny > 1 else 1000.0

    rng = np.random.default_rng(seed)
    white = rng.normal(size=(ny, nx))
    spec = np.fft.rfft2(white)

    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
    kx = kx[: spec.shape[1]] if nx % 2 == 0 else kx[: spec.shape[1]]
    kx_full = np.fft.rfftfreq(nx, d=dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx_full, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)

    k0 = 2 * np.pi / corr_length_m
    K_eff = np.maximum(K, k0)
    amp = K_eff ** (-beta / 2.0)
    amp[0, 0] = 0.0  # remove DC

    filtered = spec * amp
    surface = np.fft.irfft2(filtered, s=(ny, nx))
    surface -= surface.mean()
    current_rms = surface.std()
    if current_rms > 0:
        surface *= rms_nt / current_rms
    return surface


def region_seed(lat0_deg, lon0_deg):
    """Deterministic integer seed from a region's reference lat/lon."""
    key = round(lat0_deg * 100) * 100000 + round(lon0_deg * 100)
    return int(abs(key)) % (2 ** 31 - 1)
