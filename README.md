# fieldmark: GPS-denied navigation via magnetic + gravity map matching

A simulation of GPS-denied flight navigation that fuses a simulated IMU
(strapdown dead reckoning) with magnetic and gravity map-matching
measurements via an Extended Kalman Filter, to show how final position
accuracy depends on magnetometer/gravimeter noise and IMU grade.

## How it works

1. **Field maps come from real published models.**
   - Magnetic: [IGRF-14](https://www.ngdc.noaa.gov/IAGA/vmod/coeffs/igrf14coeffs.txt)
     (degree 13 Gauss coefficients, via the `ppigrf` spherical-harmonic
     synthesis implementation), plus a statistically-calibrated synthetic
     crustal-texture layer (`fieldmark/crustal_anomaly.py`). IGRF alone is
     far too smooth (wavelengths of thousands of km) for map matching --
     real magnetic navigation keys off short-wavelength crustal anomalies,
     which require a degree-790+ model (e.g. EMM2017) or a compiled survey
     grid (NOAA EMAG2) too large to embed here. The texture layer fills
     that gap with the right statistical character (FFT fractal surface,
     ~k^-3 spectral falloff), clearly separated from the real core field.
   - Gravity: [EGM2008](http://icgem.gfz-potsdam.de/) spherical-harmonic
     coefficients, downloaded directly from ICGEM and truncated to degree
     360 (~55 km half-wavelength -- `data/EGM2008_deg360.gfc`). Gravity
     *disturbance* is synthesized from scratch in `fieldmark/gravity.py`
     via a fully-normalized associated-Legendre recursion, referenced to
     the WGS84 normal ellipsoid (J2/J4/J6 correction).
2. **A `FieldRegion`** (`fieldmark/fieldmap.py`) synthesizes both fields
   once on a grid over a flight region and wraps them in bicubic splines,
   which give fast lookups *and* exact analytic gradients -- the gradient
   is what the EKF measurement Jacobian uses directly.
3. **An IMU error model** (`fieldmark/imu.py`) generates biased, noisy
   accelerometer/gyro measurements for consumer-MEMS/tactical/navigation
   grade presets from a true flight trajectory (`fieldmark/trajectory.py`).
4. **An EKF** (`fieldmark/ekf.py`) dead-reckons from the noisy IMU and
   periodically corrects against the magnetic/gravity maps, comparing a
   noisy "sensor" reading at the true position to the map value at the
   current estimate.
5. **Monte Carlo sweeps** (`fieldmark/simulate.py`) run many trials across
   sensor-noise levels / IMU grades and aggregate steady-state accuracy.

## Key finding: update rate matters as much as noise

Magnetometers are updated far more often than gravimeters in this sim (1 Hz
vs. 1/60 Hz, reflecting a gravimeter's much slower real-world effective
bandwidth due to the filtering needed to separate the gravity signal from
vehicle acceleration noise). In a fused run the faster sensor dominates
steady-state accuracy almost entirely -- so the noise sweeps isolate one
sensor at a time to show each one's own noise-accuracy curve, and a
separate matched-update-rate sweep shows the joint trade-off.

Similarly, with 1 Hz map updates, INS drift *between* corrections is tiny
for any IMU grade, so steady-state accuracy there is set mostly by sensor
noise, not IMU quality. IMU grade shows up in how fast position error grows
when aiding is unavailable -- see `drift_vs_imu_grade.png` / the "IMU
drift" tab in the web app.

## Setup

```
pip install -r requirements.txt
```

The repo already includes `data/igrf14coeffs.txt` and
`data/EGM2008_deg360.gfc` (downloaded once from NOAA/ICGEM); no further
downloads are needed.

## Usage

### CLI scripts (matplotlib)

```
# Build and plot the magnetic + gravity reference maps for a region
python scripts/build_fieldmaps.py --lat 40.0 --lon -105.0 --half-width 100

# Simulate one flight and plot trajectory + error-vs-time
python scripts/run_single_flight.py --imu-grade tactical --mag-noise 1 --grav-noise 2

# Monte Carlo sweeps: accuracy vs. mag noise, grav noise, and IMU-grade drift
python scripts/run_noise_sweep.py --n-trials 10 --out-dir results/
```

### Web dashboard (Streamlit)

```
streamlit run web/app.py
```

Interactive field maps (Plotly), single-flight runs, noise sweeps, and the
IMU-grade drift comparison, all backed by the same `fieldmark` core used by
the CLI scripts.

### Tests

```
python -m pytest tests/
```

## Caveats / simplifications

- Horizontal-plane-only kinematics (flat-Earth local tangent plane); altitude
  is assumed to come from a separate already-GPS-independent source (e.g.
  barometric altimeter), standard practice in this literature.
- The EKF is a simplified 6-state (position, speed, heading, gyro bias,
  accel bias) direct-state filter, not a full strapdown error-state INS.
- The magnetic crustal texture is statistically representative, not a
  literal published anomaly map (see above).
- EKF map matching assumes the position error stays within the field's
  locally-linear regime; a soft innovation gate (`ekf.py`) inflates the
  effective measurement noise instead of hard-rejecting outliers, which
  helps but doesn't fully replace a particle filter or correlation-search
  approach for large initial uncertainty / very sparse aiding.
