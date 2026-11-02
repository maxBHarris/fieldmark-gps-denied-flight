"""Strapdown IMU error model.

Generates noisy accelerometer/gyro measurements from a ground-truth
trajectory's true forward specific force and turn rate, using the standard
bias-instability + white-noise model (Groves, "Principles of GNSS,
Inertial, and Multisensor Integrated Navigation Systems"). Grade presets
below are representative of commonly cited MEMS/tactical/navigation-grade
IMUs -- real units vary, but the magnitudes and the qualitative "worse IMU
needs more frequent/accurate map fixes" behavior are what this simulation
is meant to show.

Only the along-track channel (forward specific force, yaw rate) is modeled
-- the EKF mechanization tracks speed and heading, not full body-frame
velocity -- and altitude is assumed to come from a separate, already-GPS-
independent source (barometric altimeter), which is standard in airborne
gravity/magnetic map-matching studies.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class IMUGrade:
    name: str
    gyro_bias_dph: float        # gyro bias instability, deg/hr
    gyro_arw_dphsqrth: float    # angle random walk, deg/sqrt(hr)
    accel_bias_ug: float        # accelerometer bias instability, micro-g
    accel_vrw_ugsqrthz: float   # velocity random walk, micro-g/sqrt(Hz)


GRADES = {
    "consumer_mems": IMUGrade("Consumer MEMS", gyro_bias_dph=360.0,
                               gyro_arw_dphsqrth=2.5, accel_bias_ug=5000.0,
                               accel_vrw_ugsqrthz=300.0),
    "tactical": IMUGrade("Tactical grade", gyro_bias_dph=5.0,
                          gyro_arw_dphsqrth=0.1, accel_bias_ug=300.0,
                          accel_vrw_ugsqrthz=50.0),
    "navigation": IMUGrade("Navigation grade", gyro_bias_dph=0.01,
                            gyro_arw_dphsqrth=0.002, accel_bias_ug=25.0,
                            accel_vrw_ugsqrthz=5.0),
}

G = 9.80665


def gyro_bias_rad_s(grade):
    return np.deg2rad(grade.gyro_bias_dph) / 3600.0


def accel_bias_mps2(grade):
    return grade.accel_bias_ug * 1e-6 * G


def gyro_noise_std_rad_s(grade, dt):
    """Discrete-time std of gyro white noise, rad/s, for a given sample dt."""
    arw_rad_sqrt_s = np.deg2rad(grade.gyro_arw_dphsqrth) / 60.0  # deg/sqrt(hr) -> rad/sqrt(s)
    return arw_rad_sqrt_s / np.sqrt(dt)


def accel_noise_std_mps2(grade, dt):
    """Discrete-time std of accelerometer white noise, m/s^2."""
    vrw_mps_sqrt_s = grade.accel_vrw_ugsqrthz * 1e-6 * G  # already per sqrt(Hz) == per sqrt(s)
    return vrw_mps_sqrt_s / np.sqrt(dt)


@dataclass
class ImuMeasurements:
    t: np.ndarray
    accel_fwd: np.ndarray    # measured forward specific force, m/s^2
    gyro: np.ndarray         # measured yaw rate, rad/s
    true_gyro_bias: float
    true_accel_fwd_bias: float


def simulate_measurements(trajectory, grade, dt, rng):
    """Corrupt the true IMU signals with bias + white noise for one trial."""
    accel_fwd_true = np.zeros_like(trajectory.t, dtype=float)  # constant true airspeed
    gyro_true = trajectory.turnrate

    b_gyro = rng.normal(0.0, gyro_bias_rad_s(grade))
    b_afwd = rng.normal(0.0, accel_bias_mps2(grade))

    n = len(trajectory.t)
    gyro_noise = rng.normal(0.0, gyro_noise_std_rad_s(grade, dt), size=n)
    afwd_noise = rng.normal(0.0, accel_noise_std_mps2(grade, dt), size=n)

    return ImuMeasurements(
        t=trajectory.t,
        accel_fwd=accel_fwd_true + b_afwd + afwd_noise,
        gyro=gyro_true + b_gyro + gyro_noise,
        true_gyro_bias=b_gyro,
        true_accel_fwd_bias=b_afwd,
    )
