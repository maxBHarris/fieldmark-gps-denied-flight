"""Interactive dashboard for the GPS-denied magnetic/gravity map-matching
navigation simulator.

Run with:
    streamlit run web/app.py
"""
import dataclasses
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import plotly.graph_objects as go
import streamlit as st

from fieldmark.fieldmap import FieldRegion
from fieldmark.trajectory import boustrophedon_survey, straight_and_level
from fieldmark.imu import GRADES
from fieldmark import ekf, simulate
from fieldmark.plotting import (
    plot_trajectory_error, plot_noise_sweep, plot_sweep_2d, plot_drift_vs_time,
)

st.set_page_config(page_title="GPS-Denied Nav Simulator", layout="wide")


@st.cache_resource(show_spinner="Synthesizing field maps from IGRF-14 + EGM2008...")
def get_field_region(lat0, lon0, half_width_km, n_grid):
    return FieldRegion(lat0, lon0, half_width_km=half_width_km, n_grid=n_grid)


def get_trajectory(kind, half_width_km, dt):
    if kind == "Survey (back-and-forth)":
        return boustrophedon_survey(half_width_km=half_width_km * 0.8, dt=dt)
    return straight_and_level(length_km=half_width_km * 1.8, dt=dt)


st.title("GPS-Denied Navigation via Magnetic + Gravity Map Matching")
st.caption(
    "INS (IMU dead reckoning) fused with magnetic (IGRF-14 + synthetic crustal texture) "
    "and gravity (EGM2008, degree 360) map matching via an Extended Kalman Filter."
)

with st.sidebar:
    st.header("Region")
    lat0 = st.number_input("Center latitude", value=40.0, min_value=-85.0, max_value=85.0)
    lon0 = st.number_input("Center longitude", value=-105.0, min_value=-180.0, max_value=180.0)
    half_width_km = st.slider("Region half-width (km)", 20, 300, 100, step=10)
    n_grid = st.select_slider("Map grid resolution", options=[61, 81, 121, 161], value=121)

    st.header("Flight")
    traj_kind = st.selectbox("Trajectory", ["Survey (back-and-forth)", "Straight leg"])
    imu_grade_name = st.selectbox("IMU grade", list(GRADES.keys()), index=1)

    st.header("Sensor noise (1-sigma)")
    mag_noise = st.select_slider(
        "Magnetometer noise (nT)",
        options=[0.1, 0.3, 1, 3, 10, 30, 100], value=1)
    grav_noise = st.select_slider(
        "Gravimeter noise (mGal)",
        options=[0.1, 0.3, 1, 3, 10, 30, 100], value=3)
    use_mag = st.checkbox("Use magnetic aiding", value=True)
    use_grav = st.checkbox("Use gravity aiding", value=True)

    st.header("Monte Carlo")
    n_trials = st.slider("Trials per point", 2, 15, 5)

fr = get_field_region(lat0, lon0, half_width_km, n_grid)
grade = GRADES[imu_grade_name]

tab_maps, tab_flight, tab_sweep, tab_drift = st.tabs(
    ["Field maps", "Single flight", "Noise sweep", "IMU drift"]
)

with tab_maps:
    st.subheader("Reference maps (what the EKF compares sensor readings against)")
    col1, col2 = st.columns(2)
    east_km = fr.east / 1000
    north_km = fr.north / 1000

    with col1:
        fig = go.Figure(data=go.Heatmap(z=fr.mag_grid, x=east_km, y=north_km,
                                         colorscale="Viridis",
                                         colorbar=dict(title="nT")))
        fig.update_layout(title="Magnetic total intensity (IGRF-14 + crustal texture)",
                           xaxis_title="East (km)", yaxis_title="North (km)",
                           yaxis=dict(scaleanchor="x"), height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Range: {fr.mag_grid.min():.1f} to {fr.mag_grid.max():.1f} nT")

    with col2:
        fig = go.Figure(data=go.Heatmap(z=fr.grav_grid, x=east_km, y=north_km,
                                         colorscale="RdBu_r",
                                         colorbar=dict(title="mGal")))
        fig.update_layout(title="Gravity disturbance (EGM2008, degree 360)",
                           xaxis_title="East (km)", yaxis_title="North (km)",
                           yaxis=dict(scaleanchor="x"), height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Range: {fr.grav_grid.min():.1f} to {fr.grav_grid.max():.1f} mGal")

    st.info(
        "The magnetic map = the real published IGRF-14 core field plus a statistically "
        "calibrated synthetic crustal-texture layer (IGRF alone is far too smooth, "
        "wavelength ~1000s of km, to localize position -- real magnetic navigation keys "
        "off short-wavelength crustal anomalies, which need a high-degree model too large "
        "to embed here). The gravity map is the real EGM2008 model, band-limited to degree "
        "360 (~55 km half-wavelength) and referenced to the WGS84 normal ellipsoid."
    )

with tab_flight:
    st.subheader("Simulate one flight")
    seed = st.number_input("Random seed", value=0, step=1)
    if st.button("Run flight", type="primary"):
        tr = get_trajectory(traj_kind, half_width_km, dt=1.0)
        cfg = ekf.EKFConfig(mag_noise_std_nt=mag_noise, grav_noise_std_mgal=grav_noise,
                             use_mag=use_mag, use_grav=use_grav)
        with st.spinner("Running IMU + EKF simulation..."):
            result, summary = simulate.run_trial(fr, tr, grade, cfg, dt=1.0, seed=int(seed))

        c1, c2, c3 = st.columns(3)
        c1.metric("Steady-state RMS error", f"{summary.tail_rms_m:.1f} m")
        c2.metric("Final error", f"{summary.final_error_m:.1f} m")
        c3.metric("Max error", f"{summary.max_error_m:.1f} m")
        st.caption(f"Map updates applied: {result.mag_updates} magnetic, {result.grav_updates} gravity")

        fig = plot_trajectory_error(result, fieldmap=fr)
        st.pyplot(fig)

with tab_sweep:
    st.subheader("Accuracy vs. sensor noise (Monte Carlo)")
    st.caption(
        "Magnetic and gravity aiding are swept in isolation (one enabled at a time) so "
        "each sensor's own noise-accuracy relationship is visible -- in a fused run, "
        "frequent (1 Hz) magnetic updates dominate infrequent (1/60 Hz, reflecting a "
        "gravimeter's slower effective bandwidth) gravity updates, which would otherwise "
        "hide gravity's sensitivity entirely."
    )
    if st.button("Run noise sweep", type="primary"):
        tr = get_trajectory(traj_kind, half_width_km, dt=1.0)
        base_cfg = ekf.EKFConfig()
        values = [0.1, 0.3, 1, 3, 10, 30, 100]

        with st.spinner(f"Running magnetic-only sweep ({len(values)} levels x {n_trials} trials)..."):
            mag_only = dataclasses.replace(base_cfg, use_mag=True, use_grav=False)
            mag_sweep = simulate.sweep_1d(fr, tr, grade, mag_only, "mag_noise_std_nt",
                                           values, dt=1.0, n_trials=n_trials)
        with st.spinner(f"Running gravity-only sweep ({len(values)} levels x {n_trials} trials)..."):
            grav_only = dataclasses.replace(base_cfg, use_mag=False, use_grav=True)
            grav_sweep = simulate.sweep_1d(fr, tr, grade, grav_only, "grav_noise_std_mgal",
                                            values, dt=1.0, n_trials=n_trials)

        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plot_noise_sweep(mag_sweep, "Magnetometer noise, 1-sigma (nT)",
                                        f"Magnetic-only aiding ({grade.name})"))
        with c2:
            st.pyplot(plot_noise_sweep(grav_sweep, "Gravimeter noise, 1-sigma (mGal)",
                                        f"Gravity-only aiding ({grade.name})"))

        with st.spinner("Running joint mag x gravity sweep (matched update rates)..."):
            small_values = [0.3, 1, 3, 10, 30]
            matched_cfg = dataclasses.replace(base_cfg, mag_update_interval_s=10.0,
                                               grav_update_interval_s=10.0)
            grid = simulate.sweep_2d(fr, tr, grade, matched_cfg, small_values, small_values,
                                      dt=1.0, n_trials=max(2, n_trials // 2))
        st.pyplot(plot_sweep_2d(grid, small_values, small_values))

with tab_drift:
    st.subheader("Unaided (dead-reckoning) drift by IMU grade")
    st.caption(
        "With map updates every 1 s, INS drift between fixes is tiny for any IMU grade, "
        "so steady-state accuracy (the 'Noise sweep' tab) is set mostly by sensor noise, "
        "not IMU quality. IMU grade instead shows up in how fast the position error grows "
        "once aiding is unavailable -- this isolates that, with zero initial uncertainty "
        "and no map updates at all, over a coasting period."
    )
    if st.button("Run drift comparison", type="primary"):
        tr = get_trajectory(traj_kind, half_width_km, dt=1.0)
        horizons_s = [30, 60, 120, 300, 600, 1200]
        with st.spinner("Simulating unaided drift for each IMU grade..."):
            drift = simulate.sweep_imu_grade_drift(tr, list(GRADES.keys()), dt=1.0,
                                                     horizons_s=horizons_s, n_trials=n_trials)
        st.pyplot(plot_drift_vs_time(drift, horizons_s))
