from pathlib import Path

from cycler import cycler
import colormaps as cmaps
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import animation
import numpy as np
from scipy import constants, signal
from IPython.display import HTML

import analysis
from basic import RunInfo, physics, Species, Variation
from basic.paths import (
    MPLSTYLE_FILE,
    FIGURES_FOLDER,
    V_FLOW_VARIATION_FOLDER
)
from .general import generalSaveFigure, plotEnergyEFieldOverTime

def _saveFigure(fig_name: str, sub_folder: Variation|str|None = None):
    if sub_folder is None:
        folder = "simulation-2D"
    else:
        if isinstance(sub_folder, Variation):
            sub_folder = sub_folder.value
        folder = f"simulation-2D/{sub_folder}"
    generalSaveFigure(fig_name, folder)

def maxEnergyVsAlphaFlowSpeed(info: RunInfo, normalize_energy: bool=True, save: bool=False):
    files = sorted(V_FLOW_VARIATION_FOLDER.glob("*.h5"))
    velocity = np.empty(len(files))
    W_E_max = np.empty(len(files))
    W_E_max_err = np.empty(len(files))
    for file_idx, filename in enumerate(files):
        velocity[file_idx] = int(filename.stem[-3:])
        with h5py.File(filename) as f:
            E_x = f['Electric Field/ex'][:]
            E_y = f['Electric Field/ey'][:]
        W_E = np.mean(E_x ** 2 + E_y ** 2, axis=(1,2)) * (constants.epsilon_0 / 2) / constants.electron_volt
        max_idx = np.argmax(W_E)
        max_range = W_E[max_idx-5:max_idx+5]
        W_E_max[file_idx] = np.mean(max_range)
        W_E_max_err[file_idx] = np.std(max_range)
    velocity = np.array(velocity)
    W_E_max = np.array(W_E_max)
    W_E_max_err = np.array(W_E_max_err)
    if normalize_energy:
        K_alpha_t0 = (info.alpha.si_mass * info.alpha.number_density * velocity ** 2 / (2 * constants.electron_volt))
        W_E_max /= K_alpha_t0
        W_E_max_err /= K_alpha_t0

    plt.style.use(MPLSTYLE_FILE)
    plt.figure()

    plt.errorbar(
        velocity, W_E_max * 1e-6, yerr=1e-6 * W_E_max_err, color="black",
        marker="p", markeredgecolor="black", markeredgewidth=1, markersize=10, ls="")
    y_min, y_max = plt.gca().get_ylim()
    plt.fill_between(
        [0.0, info.c_s * 1e-3], y1=y_min, y2=y_max,
        lw=2, color="lightgray", label="$u_{\\alpha} \\leq c_s$"
    )
    plt.xlabel("Initial flow velocity $u_{\\alpha}^{t=0}$ (km/s)")
    if normalize_energy:
        plt.ylabel("$\\max[\\langle W_E\\rangle_\\mathbf{r}]_t/K_\\alpha^{t=0}$  (1)")
    else:
        plt.ylabel("$\\max[\\langle W_E\\rangle_\\mathbf{r}]_t$  (MeV$\\,/\\,$m$^3$)")
    plt.xlim(90, 190)
    plt.ylim(y_min, y_max)
    plt.legend(fancybox=False, edgecolor="black")
    plt.tight_layout()
    if save:
        _saveFigure(
            f"max_energy_{'norm' if normalize_energy else ''}-vs-alpha_flow_velocity",
            "alpha_flow_velocity_variation"
        )

def _anglesForAlphaFlowSpeed(info: RunInfo, e_field_component: str):
    files = sorted(V_FLOW_VARIATION_FOLDER.glob("*.h5"))
    arr_theta = np.empty(len(files))
    arr_theta_err = np.empty(len(files))
    arr_theta_E = np.empty(len(files))
    arr_theta_E_err = np.empty(len(files))
    flow_velocity = np.empty(len(files))
    for file_idx, filename in enumerate(files):
        flow_velocity[file_idx] = int(filename.stem[-3:])
        with h5py.File(filename) as f:
            x = f["Grid/grid/X"][:] / info.lambda_D
            y = f["Grid/grid/Y"][:] / info.lambda_D
            E_x = f['Electric Field/ex'][:]
            E_y = f['Electric Field/ey'][:]
            time = f["Header/time"][:] * info.omega_pp
        res = analysis.fitGrowthRate(time, np.mean(E_x ** 2 + E_y ** 2, axis=(1,2)))
        assert res is not None, "What?"
        linear_idx = res[1]
        E_field = E_x[slice(*linear_idx)] if e_field_component == "x" else E_y[slice(*linear_idx)]
        k, k_err = analysis.waveVector2D(x, y, E_field)
        print(f"v = {flow_velocity[file_idx]} :: |k| = {np.linalg.norm(k):.4f}")
        arr_theta[file_idx], arr_theta_err[file_idx] = analysis.waveAngle2DFromWaveVector(k, k_err)

        arr_theta_E[file_idx], arr_theta_E_err[file_idx] = analysis.waveAngle2DFromElectricField(
            E_x[slice(*linear_idx)], E_y[slice(*linear_idx)]
        )
    return flow_velocity, arr_theta, arr_theta_err, arr_theta_E, arr_theta_E_err

def waveAngleVsAlphaFlowSpeed(info: RunInfo, e_field_component: str, save: bool=False):
    (
        flow_velocity,
        arr_theta,
        arr_theta_err,
        arr_theta_E,
        arr_theta_E_err,
    ) = _anglesForAlphaFlowSpeed(info, e_field_component)
    theory_v = np.linspace(90, 200, num=1000)
    theory_theta = np.arccos(
        info.c_s / (theory_v * 1e3),
        where=theory_v * 1e3 > info.c_s,
        out=np.zeros_like(theory_v)
    ) * 180 / np.pi
    plt.style.use(MPLSTYLE_FILE)
    default_cycler = (cycler(color=cmaps.devon.discrete(3).colors) +
                    cycler(linestyle=['-', '--', ':']))
    plt.rc('axes', prop_cycle=default_cycler)
    plt.fill_between([0.0, info.c_s * 1e-3], y1=-90, y2=90, lw=2, color="lightgray", label="$u_{\\alpha} \\leq c_s$")
    plt.plot(theory_v, theory_theta, label="$\\cos(\\theta)=c_s\\,/\\,u_{\\alpha}$", ls="solid")

    plt.errorbar(
        flow_velocity, arr_theta * 180 / np.pi,
        yerr=arr_theta_err * 180 / np.pi,
        marker="p", markeredgecolor="black", markeredgewidth=1, markersize=10,
        ls="", label="Estimate via $\\langle\\mathbf{k}\\rangle_{t,y}$", zorder=3
    )
    plt.errorbar(
        flow_velocity, arr_theta_E * 180 / np.pi, yerr=arr_theta_E_err,
        marker="o", markeredgecolor="black", markeredgewidth=1, markersize=10,
        ecolor="orange", ls="", zorder=2, elinewidth=4,
        label="Estimate via $\\langle \\mathbf{E}\\rangle_{t,\\mathbf{r}}$"
    )
    plt.xlabel("Initial velocity $u_{\\alpha}$ (km$\\,/\\,$s)")
    plt.ylabel("Wave angle $\\theta$ (deg)")
    plt.xlim(95, 190)
    plt.yticks(np.linspace(0, 75, num=6))
    plt.ylim(-2, 75)
    plt.legend(markerscale=0.8, fancybox=False, framealpha=1.0, edgecolor="black")
    if save:
        _saveFigure("wave_angle-vs-alpha_flow_velocity", "alpha_flow_velocity_variation")

def simulatedAlphaFlowSpeed(info: RunInfo, e_field_component: str, save: bool=False):
    (
        flow_velocity,
        arr_theta,
        arr_theta_err,
        arr_theta_E,
        arr_theta_E_err,
    ) = _anglesForAlphaFlowSpeed(info, e_field_component)
    theory_v = np.linspace(90, 200, num=1000)
    plt.style.use(MPLSTYLE_FILE)
    default_cycler = (cycler(color=cmaps.devon.discrete(3).colors) +
                    cycler(linestyle=['-', '--', ':']))
    plt.rc('axes', prop_cycle=default_cycler)
    plt.figure()
    plt.errorbar(
        flow_velocity, 1e-3 * info.c_s / np.cos(arr_theta),
        yerr=1e-3 * info.c_s * np.abs(np.sin(arr_theta) * arr_theta_err),
        marker="p", markeredgecolor="black", markeredgewidth=1, markersize=10,
        ls="", label="$u_{\\alpha}^{\\text{sim},\\mathbf{k}}=c_s\\,/\\,\\cos(\\theta_{\\text{sim},\\mathbf{k}})$", zorder=3)
    plt.plot(theory_v, theory_v, label="$u_{\\alpha}^\\text{sim}=u_{\\alpha}$", ls="--")
    plt.fill_between([0.0, info.c_s * 1e-3], y1=0, y2=info.c_s * 1e-3, lw=2, color="lightgray", label="$u_{\\alpha} \\leq c_s$")
    plt.errorbar(
        flow_velocity, 1e-3 * info.c_s / np.cos(arr_theta_E), yerr=arr_theta_E_err,
        marker="o", markeredgecolor="black", markeredgewidth=1, markersize=10,
        ecolor="orange", ls="", elinewidth=4,
        label="$u_{\\alpha}^{\\text{sim},\\mathbf{E}}=c_s\\,/\\,\\cos(\\theta_{\\text{sim},\\mathbf{E}})$", zorder=2)
    plt.xlim(90, 200)
    plt.ylim(90, 200)
    plt.gca().set_aspect('equal')
    handles, labels = plt.gca().get_legend_handles_labels()
    legend1 = plt.legend(handles[:2], labels[:2], loc="upper left", framealpha=0.5, edgecolor="black")
    plt.legend(handles[2:], labels[2:], loc="lower right", framealpha=0.5, edgecolor="black", markerscale=0.8)
    plt.gca().add_artist(legend1)
    plt.xlabel("Initial velocity $u_{\\alpha}$ (km/s)")
    plt.ylabel("Simulated velocity $u_{\\alpha}^\\text{sim}$ (km/s)")
    if save:
        _saveFigure("simulated_flow_velocity-vs-flow_velocity", "alpha_flow_velocity_variation")

def electricField2DSnapshot(filename: Path, info: RunInfo, time: float|int, save: bool=False):
    with h5py.File(filename) as f:
        if isinstance(time, float):
            time_step = np.argmin(np.abs(f["Header/time"][:] * info.omega_pp - time))
        else:
            assert abs(time) < f["Header/time"].size, "Time out of range"
            time_step = time
        x = f["Grid/x_px/Alphas/X"][0] / info.lambda_D
        y = f["Grid/y_px/Alphas/Y"][0] / info.lambda_D
        E_x = f['Electric Field/ex'][time_step]
        E_y = f['Electric Field/ey'][time_step]
    E_x_max = np.max(np.abs(E_x))
    plt.style.use(MPLSTYLE_FILE)
    plt.figure()
    plt.pcolormesh(x, y, E_x.T, cmap="bwr", vmin=-E_x_max, vmax=E_x_max)
    plt.xlabel("Position x$\\,/\\,\\lambda_D$ (1)")
    plt.ylabel("Position y$\\,/\\,\\lambda_D$ (1)")
    plt.gca().set_aspect('equal')
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(label="Electric field E$_x$ (V/m)", cax=cax)
    plt.tight_layout()
    if save:
        _saveFigure(f"electric_field_2d-idx_t={time_step}", "alpha_flow_velocity_variation")

def energyEFieldOverTime(filename: Path, info: RunInfo, save: bool=False):
    with h5py.File(filename) as f:
        E_x = f['Electric Field/ex'][:]
        E_y = f['Electric Field/ey'][:]
        time = f["Header/time"][:] * info.omega_pp
    energy = np.mean(E_x ** 2 + E_y ** 2, axis=(1,2)) * (constants.epsilon_0 / 2) / constants.electron_volt
    plt.style.use(MPLSTYLE_FILE)
    plt.figure()
    plotEnergyEFieldOverTime(time, energy, False)
    plt.ylim(1e4, 4e6)
    if save:
        _saveFigure(f"electric_field-vs-time", "alpha_flow_velocity_variation")

def strengthBFieldOverTime(filename: Path, info: RunInfo, save: bool=False):
    with h5py.File(filename) as f:
        time = f["Header/time"][1:] * info.omega_pp
        B_x = f['Magnetic Field/bx'][1:]
        B_y = f['Magnetic Field/by'][1:]
    plt.style.use(MPLSTYLE_FILE)
    plt.figure()
    plt.plot(time, np.mean(np.sqrt(B_x ** 2 + B_y ** 2) * 1e12, axis=(1,2)), lw=1, color="black")
    plt.xlabel("Time $t\\,\\omega_{pp}$ (1)")
    plt.ylabel("Magnetic field strength $\\langle\\|\\mathbf{B}\\|\\rangle_\\mathbf{r}$ (pT)")
    plt.xlim(0,150)
    if save:
        _saveFigure(f"magnetic_field_strength-vs-time", "alpha_flow_velocity_variation")

def psdBField(filename: Path, info: RunInfo, save: bool=False):
    with h5py.File(filename) as f:
        time = f["Header/time"][1:] * info.omega_pp
        B_x = f['Magnetic Field/bx'][1:]
        B_y = f['Magnetic Field/by'][1:]
    f, Pxx_bx = signal.periodogram(np.mean(B_x, axis=(1,2)), fs=1 / (time[1] - time[0]), detrend=False)
    _, Pxx_by = signal.periodogram(np.mean(B_y, axis=(1,2)), fs=1 / (time[1] - time[0]), detrend=False)

    plt.style.use(MPLSTYLE_FILE)
    plt.figure()
    plt.plot(f, Pxx_bx * 1e24, lw=1, label="PSD$[\\langle B_x\\rangle_\\mathbf{r}]$")
    plt.plot(f, Pxx_by * 1e24, lw=1, label="PSD$[\\langle B_y\\rangle_\\mathbf{r}]$")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Frequency $f\\,/\\,\\omega_{pp}$ (1)")
    plt.ylabel("PSD$[\\langle B_i\\rangle_\\mathbf{r}]$ (pT$^2$)")
    plt.legend()
    plt.xlim(1 / abs(time[-1]- time[0]), 0.5 / (time[1] - time[0]))
    if save:
        _saveFigure(f"magnetic_field_psd", "alpha_flow_velocity_variation")

def energyBField(filename: Path, info: RunInfo, save: bool=False):
    with h5py.File(filename) as f:
        time = f["Header/time"][1:] * info.omega_pp
        B_x = f['Magnetic Field/bx'][1:]
        B_y = f['Magnetic Field/by'][1:]
    energy = 0.5 * np.mean(B_x ** 2 + B_y ** 2, axis=(1,2)) / (constants.mu_0 * constants.electron_volt)

    plt.style.use(MPLSTYLE_FILE)
    plt.figure()
    plt.plot(time[1:], energy[1:], color="black", lw=1)
    plt.xlabel("Time $t\\,\\omega_{pp}$ (1)")
    plt.ylabel("Energy density $\\langle W_B\\rangle_\\mathbf{r}$ (eV$\\,/\\,$m$^3$) ")
    plt.xlim(0, 150.0)
    if save:
        _saveFigure(f"magnetic_field_energy-vs-time", "alpha_flow_velocity_variation")

def omegaVsAlphaFlowSpeed(info: RunInfo, e_field_component: str, save: bool=False):
    assert e_field_component in ["x", "y"], "Unknown e-field component"
    files = sorted(V_FLOW_VARIATION_FOLDER.glob("*.h5"))
    arr_omega = np.empty(len(files))
    arr_omega_err = np.empty(len(files))
    flow_velocity = np.empty(len(files))
    for file_idx, filename in enumerate(files):
        flow_velocity[file_idx] = int(filename.stem[-3:])
        with h5py.File(filename) as f:
            x = f["Grid/grid/X"][:] / info.lambda_D
            y = f["Grid/grid/Y"][:] / info.lambda_D
            E_x = f['Electric Field/ex'][1:]
            E_y = f['Electric Field/ey'][1:]
            time = f["Header/time"][1:] * info.omega_pp
        res = analysis.fitGrowthRate(time, np.mean(E_x ** 2 + E_y ** 2, axis=(1,2)))
        linear_idx = slice(res[1][-1])
        E_field = E_x[linear_idx] if e_field_component == "x" else E_y[linear_idx]
        arr_omega[file_idx], arr_omega_err[file_idx] = analysis.estimateFrequency(
            -3, time, E_field, n_spatial_dims=2
        )
    plt.style.use(MPLSTYLE_FILE)
    plt.figure()
    plt.errorbar(
        flow_velocity, arr_omega, yerr=arr_omega_err,
        marker="p", color="black", markersize=10, ls=""
    )
    plt.ylim(bottom=0)
    low, high = plt.gca().get_ylim()
    plt.fill_between(
        [0.0, info.c_s * 1e-3], y1=low, y2=high,
        lw=2, color="lightgray", label="$u_{\\alpha} \\leq c_s$"
    )
    plt.xlim(95, 185)
    plt.ylim(low, high)
    plt.xlabel(f"Initial flow velocity $u_\\alpha^{{t=0}}$ (km$\\,/\\,$s)")
    plt.ylabel(f"Freqency $\\omega\\,/\\,\\omega_{{pp}}$ (1)")
    if save:
        _saveFigure(f"omega-vs-alpha_flow_velocity", "alpha_flow_velocity_variation")

def psdOmegaForAlphaFlowSpeed(info: RunInfo, e_field_component: str, save: bool=False):
    assert e_field_component in ["x", "y"], "Unknown e-field component"
    files = sorted(V_FLOW_VARIATION_FOLDER.glob("*.h5"))
    flow_velocity = np.empty(len(files))
    plt.style.use(MPLSTYLE_FILE)
    plt.figure()
    for file_idx, filename in enumerate(files):
        flow_velocity[file_idx] = int(filename.stem[-3:])
        with h5py.File(filename) as f:
            x = f["Grid/grid/X"][:] / info.lambda_D
            y = f["Grid/grid/Y"][:] / info.lambda_D
            E_x = f['Electric Field/ex'][1:]
            E_y = f['Electric Field/ey'][1:]
            time = f["Header/time"][1:] * info.omega_pp
        res = analysis.fitGrowthRate(time, np.mean(E_x ** 2 + E_y ** 2, axis=(1,2)))
        linear_idx = slice(res[1][-1])
        E_field = E_x[linear_idx] if e_field_component == "x" else E_y[linear_idx]
        f, p = signal.periodogram(E_field, axis=-3, fs=1/(time[1] - time[0]))
        p_mean = np.mean(p, axis=(1,2))
        plt.plot(
            f[1:], p_mean[1:], label=f"$u_\\alpha^{{t=0}}$={int(flow_velocity[file_idx])}"
        )
    plt.xlabel("Frequency $f\\,/\\,\\omega_{pp}$ (1)")
    plt.ylabel(
        f"$\\langle\\text{{PSD}}[E_{e_field_component}]\\rangle_\\mathbf{{r}}$ $(V^2/m^2)$"
    )
    plt.legend(
        title=f"Initial flow velocity (km$\\,/\\,$s)",
        ncols=2, labelspacing=.2, columnspacing=1
    )
    plt.xlim(f[1], 0.5)
    plt.ylim(bottom=2e-4, top=2e-1)
    plt.yscale("log")
    if save:
        _saveFigure(f"psd_omega-vs-freq-vs-alpha_flow_velocity", "alpha_flow_velocity_variation")

def temperature3DOverTimeForAlphaFlowSpeed(info: RunInfo, species: Species, save: bool=False):
    plt.style.use(MPLSTYLE_FILE)
    plt.figure()
    for filename in sorted(V_FLOW_VARIATION_FOLDER.glob("*.h5")):
        v = int(filename.stem[-3:])
        with h5py.File(filename) as f:
            time = f["Header/time"][:] * info.omega_pp
            temp = np.mean(f[f"Derived/Temperature/{species.value}"], axis=(1,2))
        plt.plot(
            time, physics.kelvinToElectronVolt(temp),
            label=f"$u_\\alpha^{{t=0}}$ = {v}"
        )
    plt.legend(title="Flow velocity (km$\\,/\\,$s)")
    plt.xlabel("Time $t\\,\\omega_{pp}$ (1)")
    plt.ylabel(f"Temperature $T_{species.symbol()}$ (eV)")
    plt.xlim(0, 150.0)
    if save:
        _saveFigure(f"temp_3D_{species.value.lower()}-vs-time", "alpha_flow_velocity_variation")

def temperatureDifferences3DVsAlphaFlowSpeed(
    info: RunInfo, species: Species, n_points: int=10,
    normalize_temperature: bool=True, save: bool=False
):
    files = sorted(V_FLOW_VARIATION_FOLDER.glob("*.h5"))
    velocity = np.empty(len(files))
    T_diff = np.empty(len(files))
    T_diff_err = np.empty(len(files))
    for file_idx, filename in enumerate(files):
        velocity[file_idx] = int(filename.stem[-3:])
        with h5py.File(filename) as f:
            time = f["Header/time"][:] * info.omega_pp
            temp = physics.kelvinToElectronVolt(
                np.mean(f[f"Derived/Temperature/{species.value}"], axis=(1,2))
            )
        T_diff[file_idx] = np.mean(temp[-n_points:]) - np.mean(temp[:n_points])
        T_diff_err[file_idx] = np.sqrt(
            np.var(temp[-n_points:]) + np.var(temp[:n_points])
        ) / np.sqrt(n_points)
    if normalize_temperature:
        K_alpha_t0 = (info.alpha.si_mass * velocity ** 2 / (2 * constants.electron_volt))
        T_diff /= K_alpha_t0
        T_diff_err /= K_alpha_t0
    plt.style.use(MPLSTYLE_FILE)
    plt.figure()
    plt.errorbar(
        velocity, T_diff, yerr=T_diff_err, marker="p", color="black", markersize=10, ls=""
    )
    low, high = plt.gca().get_ylim()
    plt.fill_between([0.0, info.c_s * 1e-3], y1=low, y2=high, lw=2, color="lightgray", label="$u_{\\alpha} \\leq c_s$")
    plt.xlim(95, 185)
    plt.ylim(low, high)
    plt.xlabel(f"Initial flow velocity $u_\\alpha^{{t=0}}$ (km$\\,/\\,$s)")
    if normalize_temperature:
        plt.ylabel(f"Temperature $\\Delta T_{species.symbol()}\\,/\\,K_\\alpha^{{t=0}}$ (1)")
    else:
        plt.ylabel(f"Temperature $\\Delta T_{species.symbol()}$ (eV)")
    plt.legend(loc="lower right")
    if save:
        _saveFigure(f"temp_3D_{species.value.lower()}-vs-alpha_flow_velocity", "alpha_flow_velocity_variation")

def psdFlowVelocity(info: RunInfo, species: Species, space_dir: str, mom_dir: str, save: bool=False):
    assert space_dir in ["x", "y"], "Unknown e-field component"
    assert mom_dir in ["x", "y"], "Unknown e-field component"
    plt.style.use(MPLSTYLE_FILE)
    plt.figure()
    for filename in sorted(V_FLOW_VARIATION_FOLDER.glob("*.h5")):
        u_alpha = int(filename.stem[-3:])
        with h5py.File(filename) as f:
            time = f["Header/time"][:] * info.omega_pp
            x_grid = f[f"Grid/{space_dir}_p{mom_dir}/{species.value}/{space_dir.upper()}"][:]
            px_grid = f[f"Grid/{space_dir}_p{mom_dir}/{species.value}/P{mom_dir}"][:]
            dist_x_px = f[f'dist_fn/{space_dir}_p{mom_dir}/{species.value}'][:]

        u_x = analysis.flowVelocity1D(x_grid, px_grid, dist_x_px, info[species])
        f, Pxx = signal.periodogram(x=np.mean(u_x, axis=1), fs=1 / (0.1))
        plt.plot(f[1:], Pxx[1:], label=f"$u_\\alpha^{{t=0}}$={u_alpha}")
        if species == Species.ELECTRON:
            max_idx = np.argmax(Pxx[1:]) + 1
            print(f"{f[max_idx]} +- {f[max_idx] - f[max_idx-1]}")
            print(f[-1])
            break
    plt.xlabel("Frequency $f\\,/\\,\\omega_{pp}$ (1)")
    plt.ylabel(f"PSD$[\\langle u_{{e,{mom_dir}}}\\rangle_{space_dir}]$ (km$^2\\,/\\,$s$^2$)")
    plt.yscale("log")
    # plt.xscale("log")
    plt.legend(
        title=f"Flow velocity (km$\\,/\\,$s)",
        labelspacing=.2, columnspacing=1, ncols=2,
    )
    plt.xlim(1 / 150, 5)
    if save:
        _saveFigure(f"psd-flow_velocity-{species.value.lower()}", "alpha_flow_velocity_variation")

def flowVelocityVsTime(info: RunInfo, species: Species, space_dir: str, mom_dir: str, save: bool=False):
    assert space_dir in ["x", "y"], "Unknown e-field component"
    assert mom_dir in ["x", "y"], "Unknown e-field component"
    plt.style.use(MPLSTYLE_FILE)
    plt.figure()
    for filename in sorted(V_FLOW_VARIATION_FOLDER.glob("*.h5")):
        u_alpha = int(filename.stem[-3:])
        with h5py.File(filename) as f:
            time = f["Header/time"][:] * info.omega_pp
            x_grid = f[f"Grid/{space_dir}_p{mom_dir}/{species.value}/{space_dir.upper()}"][:]
            px_grid = f[f"Grid/{space_dir}_p{mom_dir}/{species.value}/P{mom_dir}"][:]
            dist_x_px = f[f'dist_fn/{space_dir}_p{mom_dir}/{species.value}'][:]

        u_x = analysis.flowVelocity1D(x_grid, px_grid, dist_x_px, info[species])
        plt.plot(time, np.mean(u_x, axis=1) / (u_alpha * 1e3), label=f"$u_\\alpha^{{t=0}}$={u_alpha}")
        if species == Species.ELECTRON:
            break
    plt.xlabel("Time $t\\,\\omega_{pp}$ (1)")
    plt.ylabel(f"Flow velocity $u_{{{species.symbol()},{mom_dir}}}\\,/\\,u_\\alpha^{{t=0}}$ (1)")
    plt.legend(title="Flow velocity (km$\\,/\\,$s)", labelspacing=.2)
    plt.xlim(0, 150)
    if save:
        _saveFigure(f"flow_velocity-vs-time-{species.value.lower()}", "alpha_flow_velocity_variation")

def energiesOverTime(filename: Path, info: RunInfo, save: bool=False, save_folder: Variation|str|None=None):
    W_species = []
    for species in Species:
        u_alpha = int(filename.stem[-3:])
        with h5py.File(filename) as f:
            time = f["Header/time"][:] * info.omega_pp
            x_grid = f[f"Grid/x_px/{species.value}/X"][:]
            px_grid = f[f"Grid/x_px/{species.value}/Px"][:]
            py_grid = f[f"Grid/x_py/{species.value}/Py"][:]
            dist_x_px = f[f'dist_fn/x_px/{species.value}'][:]
            dist_x_py = f[f"dist_fn/x_py/{species.value}"][:]
            T = f[f"Derived/Temperature/{species.value}"][:]
        u_x = analysis.flowVelocity1D(x_grid, px_grid, dist_x_px, info[species])
        u_y = analysis.flowVelocity1D(x_grid, py_grid, dist_x_py, info[species])
        K = 0.5 * info[species].si_mass * np.mean(u_x ** 2 + u_y ** 2, axis=1) / constants.electron_volt
        U = 3 / 2 * physics.kelvinToElectronVolt(np.mean(T, axis=(1,2)))
        W_species.append(info[species].number_density * (K + U))
    with h5py.File(filename) as f:
        time = f["Header/time"][:] * info.omega_pp
        E_x = f['Electric Field/ex'][:]
        E_y = f['Electric Field/ey'][:]
    W_E = np.mean(E_x ** 2 + E_y ** 2, axis=(1,2)) * (constants.epsilon_0 / 2) / constants.electron_volt
    W_total = W_E + np.sum(W_species, axis=0)
    plt.figure()
    for W_s, species in zip(W_species, Species):
        plt.plot(time, W_s * 1e-6, label=f"$W_{{{species.symbol()}}}$")
    plt.plot(time[1:], W_E[1:] * 1e-6, label="$W_E$")
    plt.plot(time[1:], W_total[1:] * 1e-6, label="$W_\\text{total}$")
    plt.yscale("log")
    plt.xlabel("Time $t\\,\\omega_{pp}$ (1)")
    plt.ylabel("Energy density W (MeV$\\,/\\,$m$^3$)")
    plt.xlim(0.0, 150.0)
    plt.legend(ncols=3, labelspacing=.2, columnspacing=0.5)
    if save:
        _saveFigure(f"energies-vs-time", save_folder)

def videoEFieldOverTime(
    info: RunInfo,
    filename: Path,
    direction: str,
    time_steps: range|None=None,
    label: str="",
    save: bool=False
):
    assert direction.lower() in ["x", "y"], "Direction can only be x or y"
    if save:
        assert len(label) > 1, "Label required to save animation"
    with h5py.File(filename) as f:
        time = f["Header/time"][:] * info.omega_pp
        if time_steps is None:
            time_steps = range(0, time.size, time.size // (30 * 5))
        time = time[time_steps]
        x = f["Grid/grid/X"][:] / info.lambda_D
        y = f["Grid/grid/Y"][:] / info.lambda_D
        E_field = np.transpose(f[f'Electric Field/e{direction.lower()}'][time_steps], axes=(0,2,1))
    E_x_max = np.max(np.abs(E_field)) * 0.9

    plt.style.use(MPLSTYLE_FILE)
    fig, ax = plt.subplots()
    quad = ax.pcolormesh(x, y, E_field[0], cmap="bwr", vmin=-E_x_max, vmax=E_x_max)
    text = ax.text(
        0.95, 0.95,
        horizontalalignment='right',
        verticalalignment='top',
        s=f"t$\\,\\omega_{{pp}}=\\,${time[0]:>5.1f}",
        transform=ax.transAxes
    )
    ax.set(
        xlabel = "Position x$\\,/\\,\\lambda_D$ (1)",
        ylabel = "Position y$\\,/\\,\\lambda_D$ (1)",
        aspect = "equal"
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(quad, label=f"Electric field E$_{direction.lower()}$ (V/m)", cax=cax)

    tight_bbox = fig.get_tightbbox()
    fig.set_size_inches(tight_bbox.width, tight_bbox.height)
    fig.set_layout_engine("tight", pad=0.01)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    def update(frame_idx):
        quad.set_array(E_field[frame_idx])
        text.set_text(f"t$\\,\\omega_{{pp}}\\,=\\,${time[frame_idx]:>5.1f}")
        return text, quad

    ani = animation.FuncAnimation(fig=fig, func=update, frames=range(len(list(time_steps))))
    if save:
        folder = FIGURES_FOLDER / "simulation-2D"
        folder.mkdir(exist_ok=True, parents=True)
        ani.save(
            filename=folder / f"e_field_{direction.lower()}_evolution-{label}.gif",
            writer="ffmpeg", fps=30
        )
        plt.close()
    else:
        return HTML(ani.to_jshtml())

def _loadPxPyDistribution(
    info: RunInfo,
    species: Species,
    filename: Path,
    time: float|int|range,
    normalized_velocity: bool,
):
    with h5py.File(filename) as f:
        sim_time = f["Header/time"][:] * info.omega_pp
        if isinstance(time, range):
            t_idx = time
        else:
            t_idx = np.argmin(np.abs(sim_time - time))
        sim_time = sim_time[t_idx]
        x_grid = f[f"Grid/grid/X"][:]
        y_grid = f[f"Grid/grid/Y"][:]
        px_grid = f[f"Grid/px_py/{species}/Px"]
        if px_grid.ndim > 1:
            px_grid = f[f"Grid/px_py/{species}/Px"][t_idx]
            py_grid = f[f"Grid/px_py/{species}/Py"][t_idx]
        else:
            px_grid = f[f"Grid/px_py/{species}/Px"][:]
            py_grid = f[f"Grid/px_py/{species}/Py"][:]
        px_py = f[f'dist_fn/px_py/{species}'][t_idx]
    v_x, v_y, f_v = analysis.normalizeDistributionPxPy(
        x_grid, y_grid, px_grid, py_grid, px_py, info[species]
    )
    if normalized_velocity:
        v_x /= info[species].v_thermal
        v_y /= info[species].v_thermal
    else:
        v_x *= 1e-3
        v_y *= 1e-3
    return v_x, v_y, f_v

def videoPxPyDistribution(
    info: RunInfo,
    species: Species,
    filename: Path,
    time_steps: range|None=None,
    normalized_velocity: bool=True,
    label: str="",
    save: bool=False
):
    with h5py.File(filename) as f:
        time = f["Header/time"][:] * info.omega_pp
        if time_steps is None:
            time_steps = range(0, time.size, time.size // (30 * 5))
        time = time[time_steps]
    v_x, v_y, f_v = _loadPxPyDistribution(
        info, species, filename, time_steps, normalized_velocity
    )
    dv_x = abs(v_x[1] - v_x[0])
    dv_y = abs(v_y[1] - v_y[0])
    v_x = np.concat([[v_x[0]-dv_x], v_x]) + dv_x / 2
    v_y = np.concat([[v_y[0]-dv_y], v_y]) + dv_y / 2
    non_zero_v_x = v_x[np.nonzero(np.sum(f_v, axis=(0,2)) > 0)]
    non_zero_v_y = v_y[np.nonzero(np.sum(f_v, axis=(0,1)) > 0)]
    f_v[f_v<=0] = np.min(f_v[f_v>0])

    plt.style.use(MPLSTYLE_FILE)
    fig, ax = plt.subplots()
    quad = ax.pcolormesh(v_x, v_y, f_v[0].T, norm="log")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(quad, label=f"$\\langle f_{species.symbol()}\\rangle_\\mathbf{{r}}$ (s$\\,/\\,$m$^3$)", cax=cax)

    text = plt.text(0.95, 0.95,
            horizontalalignment='right',
            verticalalignment='top',
            s=f"t$\\,\\omega_{{pp}}=\\,${time[0]:>5.1f}",
            transform=ax.transAxes
    )
    if normalized_velocity:
        ax.set_xlabel(f"Velocity $v_{{{species.symbol()},x}}\\,/\\,v^{{t=0}}_{species.symbol()}$ (1)")
        ax.set_ylabel(f"Velocity $v_{{{species.symbol()},y}}\\,/\\,v^{{t=0}}_{species.symbol()}$ (1)")
    else:
        ax.set_ylabel(f"Velocity $v_{{{species.symbol()},x}}$ (km/s)")
        ax.set_ylabel(f"Velocity $v_{{{species.symbol()},y}}$ (km/s)")
    ax.set_xlim(np.min(non_zero_v_x), np.max(non_zero_v_x))
    ax.set_ylim(np.min(non_zero_v_y), np.max(non_zero_v_y))

    def update(frame_idx):
        quad.set_array(f_v[frame_idx].T)
        text.set_text(f"t$\\,\\omega_{{pp}}\\,=\\,${time[frame_idx]:>5.1f}")
        return (quad, text,)

    frames = list(range(len(list(time_steps))))
    from tqdm import tqdm
    ani = animation.FuncAnimation(fig=fig, func=update, frames=frames)
    if save:
        FIGURES_FOLDER.mkdir(exist_ok=True)
        ani.save(
            filename=FIGURES_FOLDER / f"px_py_distribution-{species}-{label}.mp4",
            writer="ffmpeg", fps=30
        )
        plt.close()
    else:
        return HTML(ani.to_jshtml())

def pxPyDistribution(
    info: RunInfo,
    species: Species,
    filename: Path,
    time: float|int,
    normalized_velocity: bool=True,
    save: bool=False,
):
    v_x, v_y, f_v = _loadPxPyDistribution(info, species, filename, time, normalized_velocity)
    dv_x = abs(v_x[1] - v_x[0])
    dv_y = abs(v_y[1] - v_y[0])
    v_x = np.concat([[v_x[0]-dv_x], v_x]) + dv_x / 2
    v_y = np.concat([[v_y[0]-dv_y], v_y]) + dv_y / 2
    non_zero_v_x = v_x[np.nonzero(np.sum(f_v, axis=1) > 0)]
    non_zero_v_y = v_y[np.nonzero(np.sum(f_v, axis=0) > 0)]
    f_v[f_v<=0] = np.min(f_v[f_v>0])
    plt.pcolormesh(v_x, v_y, f_v.T, norm="log")
    plt.colorbar(label=f"$\\langle f_{species.symbol()}\\rangle_\\mathbf{{r}}$ (s$\\,/\\,$m$^3$)")
    plt.xlim(np.min(non_zero_v_x), np.max(non_zero_v_x))
    plt.ylim(np.min(non_zero_v_y), np.max(non_zero_v_y))
    if normalized_velocity:
        plt.xlabel(f"Velocity $v_{{{species.symbol()},x}}\\,/\\,v^{{t=0}}_{species.symbol()}$ (1)")
        plt.ylabel(f"Velocity $v_{{{species.symbol()},y}}\\,/\\,v^{{t=0}}_{species.symbol()}$ (1)")
    else:
        plt.ylabel(f"Velocity $v_{{{species.symbol()},x}}$ (km/s)")
        plt.ylabel(f"Velocity $v_{{{species.symbol()},y}}$ (km/s)")
    if save:
        generalSaveFigure(f"px_py_distribution-{species}-t={time}")