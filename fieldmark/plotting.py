"""Matplotlib plotting helpers shared by the CLI scripts."""
import numpy as np
import matplotlib.pyplot as plt


def plot_field_maps(fieldmap):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    extent = [fieldmap.east[0] / 1000, fieldmap.east[-1] / 1000,
              fieldmap.north[0] / 1000, fieldmap.north[-1] / 1000]

    im0 = axes[0].imshow(fieldmap.mag_grid, extent=extent, origin="lower",
                          cmap="viridis", aspect="equal")
    axes[0].set_title("Magnetic total intensity (IGRF-14 + crustal texture)")
    axes[0].set_xlabel("East (km)")
    axes[0].set_ylabel("North (km)")
    fig.colorbar(im0, ax=axes[0], label="nT")

    im1 = axes[1].imshow(fieldmap.grav_grid, extent=extent, origin="lower",
                          cmap="RdBu_r", aspect="equal")
    axes[1].set_title("Gravity disturbance (EGM2008, degree 360)")
    axes[1].set_xlabel("East (km)")
    axes[1].set_ylabel("North (km)")
    fig.colorbar(im1, ax=axes[1], label="mGal")

    fig.suptitle(f"Field maps centered at {fieldmap.lat0:.3f}, {fieldmap.lon0:.3f}")
    fig.tight_layout()
    return fig


def plot_trajectory_error(result, fieldmap=None, background="mag"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    if fieldmap is not None:
        extent = [fieldmap.east[0] / 1000, fieldmap.east[-1] / 1000,
                  fieldmap.north[0] / 1000, fieldmap.north[-1] / 1000]
        grid = fieldmap.mag_grid if background == "mag" else fieldmap.grav_grid
        cmap = "viridis" if background == "mag" else "RdBu_r"
        ax.imshow(grid, extent=extent, origin="lower", cmap=cmap, alpha=0.6, aspect="equal")

    ax.plot(result.true_e / 1000, result.true_n / 1000, "k-", lw=1.5, label="True trajectory")
    ax.plot(result.est_e / 1000, result.est_n / 1000, "r--", lw=1.2, label="EKF estimate")
    ax.set_xlabel("East (km)")
    ax.set_ylabel("North (km)")
    ax.set_title("True vs. estimated trajectory")
    ax.legend(loc="best")
    ax.set_aspect("equal")

    ax2 = axes[1]
    t_min = result.t / 60.0
    ax2.plot(t_min, result.pos_error_m, color="C3", label="Position error")
    ax2.plot(t_min, result.pos_std_m, color="C0", ls="--", label=r"EKF 1$\sigma$ (pos)")
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("Error (m)")
    ax2.set_title("Position error vs. time")
    ax2.legend(loc="best")
    ax2.set_yscale("log")

    fig.tight_layout()
    return fig


def plot_noise_sweep(sweep, xlabel, title, logx=True):
    """sweep: dict value -> array of per-trial tail-RMS errors (m)."""
    values = sorted(sweep.keys())
    means = np.array([sweep[v].mean() for v in values])
    stds = np.array([sweep[v].std() for v in values])

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.errorbar(values, means, yerr=stds, marker="o", capsize=4, color="C0")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Steady-state position error, RMS (m)")
    ax.set_title(title)
    if logx:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_drift_vs_time(drift_by_grade, horizons_s):
    """drift_by_grade: dict grade_name -> {horizon_s -> array of errors (m)},
    as returned by fieldmark.simulate.sweep_imu_grade_drift."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    minutes = np.array(horizons_s) / 60.0
    for name, by_horizon in drift_by_grade.items():
        means = np.array([by_horizon[h].mean() for h in horizons_s])
        ax.plot(minutes, means, marker="o", label=name)
    ax.set_xlabel("Time since last fix (min)")
    ax.set_ylabel("Position error (m)")
    ax.set_title("Unaided (dead-reckoning) drift by IMU grade")
    ax.set_yscale("log")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_sweep_2d(grid, mag_values, grav_values):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="inferno_r",
                    extent=[0, len(mag_values), 0, len(grav_values)])
    ax.set_xticks(np.arange(len(mag_values)) + 0.5)
    ax.set_xticklabels([f"{v:g}" for v in mag_values], rotation=45)
    ax.set_yticks(np.arange(len(grav_values)) + 0.5)
    ax.set_yticklabels([f"{v:g}" for v in grav_values])
    ax.set_xlabel("Magnetometer noise (nT)")
    ax.set_ylabel("Gravimeter noise (mGal)")
    ax.set_title("Median steady-state position error (m)")
    fig.colorbar(im, ax=ax, label="m")

    for i in range(len(grav_values)):
        for j in range(len(mag_values)):
            ax.text(j + 0.5, i + 0.5, f"{grid[i, j]:.0f}", ha="center", va="center",
                    color="white", fontsize=8)

    fig.tight_layout()
    return fig
