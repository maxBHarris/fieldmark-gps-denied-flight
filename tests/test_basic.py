"""Lightweight sanity/regression checks for the core field + nav models.

Not a full test suite -- just guards against the kind of silent breakage
(wrong units, sign errors, broken gradients) that's easy to introduce when
touching the spherical-harmonic synthesis or EKF math.
"""
import numpy as np
import pytest

from fieldmark.gravity import GravityModel
from fieldmark.igrf import total_intensity_grid
from fieldmark.fieldmap import FieldRegion
from fieldmark import ekf, simulate
from fieldmark.imu import GRADES
from fieldmark.trajectory import boustrophedon_survey, from_waypoints

GFC_PATH = "data/EGM2008_deg360.gfc"


def test_igrf_total_intensity_is_realistic():
    lat = np.array([[40.0]])
    lon = np.array([[-105.0]])
    B = total_intensity_grid(lat, lon, alt_km=8.0)
    assert 25000 < B[0, 0] < 65000  # nT, realistic mid-latitude total field


def test_gravity_disturbance_is_realistic():
    gm = GravityModel(GFC_PATH, nmax=360)
    d = gm.disturbance_mgal(np.array([40.0]), np.array([-105.0]), 0.0)
    assert -500 < d[0] < 500  # mGal, realistic regional disturbance magnitude


def test_fieldmap_gradients_match_finite_difference():
    fr = FieldRegion(40.0, -105.0, half_width_km=100.0, n_grid=81)
    e, n = 5000.0, -3000.0
    eps = 50.0

    vnum = (fr.mag_value(e + eps, n) - fr.mag_value(e - eps, n)) / (2 * eps)
    de, _ = fr.mag_grad(e, n)
    assert de == pytest.approx(vnum, rel=1e-3)

    vnum_g = (fr.grav_value(e + eps, n) - fr.grav_value(e - eps, n)) / (2 * eps)
    deg, _ = fr.grav_grad(e, n)
    assert deg == pytest.approx(vnum_g, rel=1e-3)


def test_map_matching_beats_unaided_dead_reckoning():
    fr = FieldRegion(40.0, -105.0, half_width_km=100.0, n_grid=81)
    tr = boustrophedon_survey(half_width_km=80, leg_spacing_km=20, speed_mps=180, dt=1.0)
    grade = GRADES["tactical"]

    unaided_cfg = ekf.EKFConfig(use_mag=False, use_grav=False)
    _, unaided = simulate.run_trial(fr, tr, grade, unaided_cfg, dt=1.0, seed=0)

    aided_cfg = ekf.EKFConfig(mag_noise_std_nt=1.0, grav_noise_std_mgal=2.0)
    _, aided = simulate.run_trial(fr, tr, grade, aided_cfg, dt=1.0, seed=0)

    assert aided.tail_rms_m < unaided.tail_rms_m / 10


def test_from_waypoints_tracks_a_zigzag_route():
    waypoints = [(0, 0), (10000, 0), (10000, 10000), (0, 10000), (0, 20000)]
    tr = from_waypoints(waypoints, speed_mps=180.0, turn_rate_dps=5.0, dt=1.0)

    assert not np.any(np.isnan(tr.east))
    assert not np.any(np.isnan(tr.north))
    assert (tr.east[0], tr.north[0]) == waypoints[0]

    for wp in waypoints[1:]:
        dist = np.hypot(tr.east - wp[0], tr.north - wp[1])
        assert dist.min() < 300.0  # within a couple of discretization steps


def test_accuracy_degrades_monotonically_with_mag_noise():
    fr = FieldRegion(40.0, -105.0, half_width_km=100.0, n_grid=81)
    tr = boustrophedon_survey(half_width_km=80, leg_spacing_km=20, speed_mps=180, dt=1.0)
    grade = GRADES["tactical"]

    errs = []
    for noise in [0.3, 3.0, 30.0]:
        c = ekf.EKFConfig(use_mag=True, use_grav=False, mag_noise_std_nt=noise)
        _, summary = simulate.run_trial(fr, tr, grade, c, dt=1.0, seed=1)
        errs.append(summary.tail_rms_m)

    assert errs[0] < errs[1] < errs[2]
