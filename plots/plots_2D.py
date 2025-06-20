from pathlib import Path

from cycler import cycler
import colormaps as cmaps
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy import constants, signal, optimize, stats
from IPython.display import HTML

import analysis
from basic import RunInfo, physics, Species, Variation
from basic.paths import (
    MPLSTYLE_FILE,
    FIGURES_FOLDER,
    FOLDER_2D,
    V_FLOW_VARIATION_FOLDER,
    THEORY_U_ALPHA_FILE,
    THEORY_DENSITY_RATIO_FILE,
)
from plots.settings import (
    MARKERS,
    FIGURE_FULL_SIZE,
    FIGURE_HALF_SIZE,
)
from .general import generalSaveFigure, plotEnergyEFieldOverTime, generalSaveVideo

def _saveFigure(fig_name: str, sub_folder: Variation|str|None=None):
    if sub_folder is None:
        folder = "simulation-2D"
    else:
        if isinstance(sub_folder, Variation):
            sub_folder = sub_folder.value
        folder = f"simulation-2D/{sub_folder}"
    generalSaveFigure(fig_name, folder)

def _saveVideo(ani: FuncAnimation, vid_name: str, sub_folder: Variation|str|None=None):
    if sub_folder is None:
        folder = "simulation-2D"
    else:
        if isinstance(sub_folder, Variation):
            sub_folder = sub_folder.value
        folder = f"simulation-2D/{sub_folder}"
    generalSaveVideo(ani, vid_name, folder)

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
    plt.xlabel("Flow velocity $u_{\\alpha}^{t=0}$ (km$\\,/\\,$s)")
    if normalize_energy:
        plt.ylabel("$\\max[\\langle W_E\\rangle_\\mathbf{r}]_t\\,/\\,K_\\alpha^{t=0}$  (1)")
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
        res = analysis.fitGrowthRate(time, np.mean(E_x ** 2 + E_y ** 2, axis=(1,2)), allowed_slope_deviation=0.5)
        assert res is not None, "What?"
        linear_idx = res[1]
        E_field = E_x[slice(*linear_idx)] if e_field_component == "x" else E_y[slice(*linear_idx)]
        k, k_err = analysis.waveVector2D(x, y, E_field)
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
    with h5py.File(THEORY_U_ALPHA_FILE) as f:
        theory_v = f["u_alpha_bulk"][:] / 1e3
        theory_theta = f["theta_max"][:] * 180 / np.pi

    plt.style.use(MPLSTYLE_FILE)
    default_cycler = (cycler(color=cmaps.devon.discrete(3).colors) +
                    cycler(linestyle=['-', '--', ':']))
    plt.rc('axes', prop_cycle=default_cycler)
    plt.figure(figsize=FIGURE_FULL_SIZE)
    # plt.fill_between([0.0, info.c_s * 1e-3], y1=-90, y2=90, lw=2, color="lightgray", label="$u_{\\alpha} \\leq c_s$")
    plt.plot(theory_v, theory_theta, label="Linear theory $\\theta_\\text{max}$", ls="solid") # $\\cos(\\theta)=c_s\\,/\\,u_{\\alpha}$

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
    plt.xlabel("Flow velocity $u_{\\alpha}$ (km$\\,/\\,$s)")
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
    plt.xlabel("Velocity $u_{\\alpha}$ (km/s)")
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
        x = f["Grid/grid/X"] / info.lambda_D
        y = f["Grid/grid/Y"] / info.lambda_D
        E_x = f['Electric Field/ex'][time_step]
    E_x_max = np.max(np.abs(E_x))
    plt.style.use(MPLSTYLE_FILE)
    plt.figure(figsize=FIGURE_FULL_SIZE)
    plt.pcolormesh(x, y, E_x.T, cmap="bwr", rasterized=True, vmin=-0.8, vmax=0.8)
    plt.xlabel("Position x$\\,/\\,\\lambda_\\text{D}$ (1)")
    plt.ylabel("Position y$\\,/\\,\\lambda_\\text{D}$ (1)")
    plt.xticks(np.linspace(0, 64, num=5))
    plt.yticks(np.linspace(0, 32, num=5))
    plt.gca().set_aspect('equal')
    divider = make_axes_locatable(plt.gca())
    cax: plt.Axes = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(label="Electric field E$_x$ (V/m)", cax=cax)
    cax.set_yticks(np.linspace(-0.8, 0.8, num=5))
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
    plt.xlabel("Time $t\\,\\omega_\\text{pp}$ (1)")
    plt.ylabel("Magnetic field strength $\\langle\\|\\mathbf{B}\\|\\rangle_\\mathbf{r}$ (pT)")
    plt.xlim(0,150)
    if save:
        _saveFigure(f"magnetic_field_strength-vs-time", "alpha_flow_velocity_variation")

def psdBField(filename: Path, info: RunInfo, save: bool=False):
    with h5py.File(filename) as f:
        time = f["Header/time"][1:] * info.omega_pp
        B_x = f['Magnetic Field/bx'][1:]
        B_y = f['Magnetic Field/by'][1:]
    f, Pxx_bx = signal.periodogram(np.mean(B_x, axis=(1,2)), fs=1 / (time[1] - time[0]), detrend=False, scaling="spectrum")
    _, Pxx_by = signal.periodogram(np.mean(B_y, axis=(1,2)), fs=1 / (time[1] - time[0]), detrend=False, scaling="spectrum")

    plt.style.use(MPLSTYLE_FILE)
    plt.figure(figsize=(FIGURE_HALF_SIZE[0], 2.7))
    plt.plot(f, Pxx_bx * 1e18, lw=1, label="PSD$[\\langle B_x\\rangle_\\mathbf{r}]$")
    plt.plot(f, Pxx_by * 1e18, lw=1, label="PSD$[\\langle B_y\\rangle_\\mathbf{r}]$")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Frequency $f\\,/\\,\\omega_\\text{pp}$ (1)")
    plt.ylabel("PSD$[\\langle B_i\\rangle_\\mathbf{r}]$ (nT$^2$)")
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
    plt.figure(figsize=(FIGURE_HALF_SIZE[0], 2.7))
    plt.plot(time[1:], energy[1:], color="black", lw=1)
    plt.xlabel("Time $t\\,\\omega_\\text{pp}$ (1)")
    plt.ylabel("Energy density $\\langle W_B\\rangle_\\mathbf{r}$ (eV$\\,/\\,$m$^3$) ")
    plt.xlim(0, 150.0)
    if save:
        _saveFigure(f"magnetic_field_energy-vs-time", "alpha_flow_velocity_variation")

def omegaVsAlphaFlowSpeed(info: RunInfo, save: bool=False):
    files = sorted(V_FLOW_VARIATION_FOLDER.glob("*.h5"))
    arr_omega_x = np.empty(len(files))
    arr_omega_x_err = np.empty(len(files))
    arr_omega_y = np.empty(len(files))
    arr_omega_y_err = np.empty(len(files))
    flow_velocity = np.empty(len(files))
    for file_idx, filename in enumerate(files):
        flow_velocity[file_idx] = int(filename.stem[-3:])
        with h5py.File(filename) as f:
            E_x = f['Electric Field/ex'][1:]
            E_y = f['Electric Field/ey'][1:]
            time = f["Header/time"][1:] * info.omega_pp
        res = analysis.fitGrowthRate(time, np.mean(E_x ** 2 + E_y ** 2, axis=(1,2)), allowed_slope_deviation=0.5)
        linear_idx = slice(res[1][-1])
        arr_omega_x[file_idx], arr_omega_x_err[file_idx] = analysis.estimateFrequency(
            -3, time, E_x[linear_idx], n_spatial_dims=2
        )
        arr_omega_y[file_idx], arr_omega_y_err[file_idx] = analysis.estimateFrequency(
            -3, time, E_y[linear_idx], n_spatial_dims=2
        )
    with h5py.File(THEORY_U_ALPHA_FILE) as f:
        theory_v = f["u_alpha_bulk"][:] / 1e3
        theory_omega = f["omega_max"][:] / info.omega_pp
    plt.style.use(MPLSTYLE_FILE)
    default_cycler = (cycler(color=cmaps.devon.discrete(3).colors) +
                    cycler(linestyle=['-', '--', ':']))
    plt.rc('axes', prop_cycle=default_cycler)
    plt.figure(figsize=(FIGURE_HALF_SIZE[0], 2.5))
    plt.plot(theory_v, theory_omega, ls="solid", label="Linear theory")
    plt.errorbar(
        flow_velocity, arr_omega_x, yerr=arr_omega_x_err,
        marker="o", ls="", color="white",
        markeredgecolor="black", markeredgewidth=1,
        label=r"Sim. $\omega_{\text{max},E_x}$", ecolor="black", elinewidth=1.5,
        zorder=4,
    )
    plt.errorbar(
        flow_velocity, arr_omega_y, yerr=arr_omega_y_err,
        marker="v", ls="",
        markeredgecolor="black", markeredgewidth=1,
        label=r"Sim. $\omega_{\text{max},E_y}$", ecolor="black", elinewidth=1.5,
    )
    plt.ylim(0.2, 0.9)
    low, high = plt.gca().get_ylim()
    plt.xlim(95, 185)
    plt.ylim(low, high)
    plt.xlabel(f"Flow velocity $u_\\alpha^{{t=0}}$ (km$\\,/\\,$s)")
    plt.ylabel(f"Frequency $\\omega_\\text{{max}}\\,/\\,\\omega_\\text{{pp}}$ (1)")
    plt.legend(loc=(0.04, 0.2), labelspacing=0.4)
    if save:
        _saveFigure(f"omega-vs-alpha_flow_velocity", "alpha_flow_velocity_variation")

def wavenumberVsAlphaFlowSpeed(info: RunInfo, save: bool=False):
    files = sorted(V_FLOW_VARIATION_FOLDER.glob("*.h5"))
    arr_kx = np.empty(len(files))
    arr_kx_err = np.empty(len(files))
    arr_ky = np.empty(len(files))
    arr_ky_err = np.empty(len(files))
    flow_velocity = np.empty(len(files))
    for file_idx, filename in enumerate(files):
        flow_velocity[file_idx] = int(filename.stem[-3:])
        with h5py.File(filename) as f:
            x = f["Grid/grid/X"][:] / info.lambda_D
            y = f["Grid/grid/Y"][:] / info.lambda_D
            E_x = f['Electric Field/ex'][1:]
            E_y = f['Electric Field/ey'][1:]
            time = f["Header/time"][1:] * info.omega_pp
        res = analysis.fitGrowthRate(
            time, np.mean(E_x ** 2 + E_y ** 2, axis=(1,2)),
            allowed_slope_deviation=0.5
        )
        linear_idx = slice(res[1][-1])
        k, k_err = analysis.waveVector2D(x, y, E_x[linear_idx])
        arr_kx[file_idx] = np.linalg.norm(k)
        arr_kx_err[file_idx] = np.linalg.norm(k * k_err / np.linalg.norm(k))
        k, k_err = analysis.waveVector2D(x, y, E_y[linear_idx])
        arr_ky[file_idx] = np.linalg.norm(k)
        arr_ky_err[file_idx] = np.linalg.norm(k * k_err / np.linalg.norm(k))

    with h5py.File(THEORY_U_ALPHA_FILE) as f:
        theory_v = f["u_alpha_bulk"][:] / 1e3
        theory_k = f["k_max"][:] * info.lambda_D

    plt.style.use(MPLSTYLE_FILE)
    default_cycler = (cycler(color=cmaps.devon.discrete(3).colors) +
                    cycler(linestyle=['-', '--', ':']))
    plt.rc('axes', prop_cycle=default_cycler)
    plt.figure(figsize=(FIGURE_HALF_SIZE[0], 2.5))
    plt.plot(theory_v, theory_k, ls="solid", label="Linear theory")
    plt.errorbar(
        flow_velocity, arr_kx, yerr=arr_kx_err,
        marker="o", ls="", color="white",
        markeredgecolor="black", markeredgewidth=1, zorder=4,
        label=r"Sim. $k_{\text{max},E_x}$", ecolor="black", elinewidth=1.5
    )
    plt.errorbar(
        flow_velocity, arr_ky, yerr=arr_ky_err,
        marker="v", ls="",
        markeredgecolor="black", markeredgewidth=1,
        label=r"Sim. $k_{\text{max},E_y}$", ecolor="black", elinewidth=1.5
    )
    plt.ylim(bottom=0.2)
    low, high = plt.gca().get_ylim()
    plt.xlim(95, 185)
    plt.ylim(low, high)
    plt.xlabel(f"Flow velocity $u_\\alpha^{{t=0}}$ (km$\\,/\\,$s)")
    plt.ylabel(f"Wave number $k_\\text{{max}}\\,\\lambda_\\text{{D}}$ (1)")
    plt.legend(loc=(0.05, 0.05), labelspacing=0.4)
    if save:
        _saveFigure(f"k-vs-alpha_flow_velocity", "alpha_flow_velocity_variation")

def psdOmegaForAlphaFlowSpeed(info: RunInfo, e_field_component: str, save: bool=False):
    assert e_field_component in ["x", "y"], "Unknown e-field component"
    files = sorted(V_FLOW_VARIATION_FOLDER.glob("*.h5"))
    flow_velocity = np.empty(len(files))
    plt.style.use(MPLSTYLE_FILE)
    plt.figure(figsize=FIGURE_FULL_SIZE)
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
        f *= 2 * np.pi
        def lorentzian( x, x0, gam, a, b):
            return a * gam**2 / ( gam**2 + ( x - x0 )**2) + b

        popt, pcov = optimize.curve_fit(
            lambda x, x0, gam, a, b: np.log(lorentzian(x, x0, gam, a, b)),
            f[f<np.pi][1:], np.log(p_mean[f<np.pi][1:]),
            p0=[f[1:][np.argmax(p_mean[1:])], 0.05, np.max(p_mean[1:]), 0])
        fit_f = np.linspace(0, np.pi, num=100)
        # plt.plot(fit_f, lorentzian(fit_f, *popt), color="black", lw=1)
        plt.plot(
            f[1:], p_mean[1:], label=f"$u_\\alpha^{{t=0}}$={int(flow_velocity[file_idx])}"
        )
    plt.xlabel("Frequency $\\omega\\,/\\,\\omega_\\text{pp}$ (1)")
    plt.ylabel(
        f"$\\langle\\text{{PSD}}[E_{e_field_component}]\\rangle_\\mathbf{{r}}$ (V$^2\\,/\\,$m$^2$)"
    )
    plt.legend(
        title=f"Flow velocity (km$\\,/\\,$s)",
        ncols=2, labelspacing=.2, columnspacing=1
    )
    plt.xlim(0, np.pi)
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
    plt.legend(title="Flow velocity (km$\\,/\\,$s)", labelspacing=.4)
    plt.xlabel("Time $t\\,\\omega_\\text{pp}$ (1)")
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
            temp = physics.kelvinToElectronVolt(
                np.mean(f[f"Derived/Temperature/{species.value}"], axis=(1,2))
            )
        T_diff[file_idx] = np.mean(temp[-n_points:]) - np.mean(temp[:n_points])
        T_diff_err[file_idx] = np.sqrt(
            np.var(temp[-n_points:]) + np.var(temp[:n_points])
        ) / np.sqrt(n_points)
    if normalize_temperature:
        K_alpha_t0 = (info.alpha.si_mass * (velocity * 1e3) ** 2 / (2 * constants.electron_volt))
        T_diff /= K_alpha_t0
        T_diff_err /= K_alpha_t0
    plt.style.use(MPLSTYLE_FILE)
    plt.figure(figsize=(FIGURE_HALF_SIZE[0], 2.5))
    plt.errorbar(
        velocity, T_diff, yerr=T_diff_err, marker="o", color="black", markersize=10, ls=""
    )
    low, high = plt.gca().get_ylim()
    plt.fill_between([0.0, info.c_s * 1e-3], y1=low, y2=high, lw=2, color="lightgray", label="$u_{\\alpha} \\leq c_s$")
    plt.xlim(95, 185)
    plt.ylim(low, high)
    plt.xlabel(f"Flow velocity $u_\\alpha^{{t=0}}$ (km$\\,/\\,$s)")
    if normalize_temperature:
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        plt.ylabel(f"Heating $\\Delta T_{species.symbol()}\\,/\\,K_\\alpha^{{t=0}}$ (1)")
    else:
        plt.ylabel(f"Heating $\\Delta T_{species.symbol()}$ (eV)")
    plt.legend(loc="best")
    if save:
        _saveFigure(
            f"temp_3D_{species.value.lower()}-vs-alpha_flow_velocity", "alpha_flow_velocity_variation"
        )

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
        f *= 2 * np.pi
        plt.plot(f[1:], Pxx[1:], label=f"$u_\\alpha^{{t=0}}$={u_alpha}")
        if species == Species.ELECTRON:
            max_idx = np.argmax(Pxx[1:]) + 1
            print(f"{f[max_idx]} +- {f[max_idx] - f[max_idx-1]}")
            print(f[-1])
            break
    plt.xlabel("Frequency $\\omega\\,/\\,\\omega_\\text{pp}$ (1)")
    plt.ylabel(f"PSD$[\\langle u_{{e,{mom_dir}}}\\rangle_{space_dir}]$ (km$^2\\,/\\,$s$^2$)")
    plt.yscale("log")
    # plt.xscale("log")
    plt.legend(
        title=f"Flow velocity (km$\\,/\\,$s)",
        labelspacing=.2, columnspacing=1, ncols=2,
    )
    plt.xlim(2 * np.pi / 150, 5 * 2 * np.pi)
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
    plt.xlabel("Time $t\\,\\omega_\\text{pp}$ (1)")
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
        print(f"{u_alpha} :: {species} | K_-1/K_0 = {K[-1] / K[0]} and U_-1/K_0 = {U[-1] / K[0]} and U_-1/U_0 = {U[-1] / U[0]}")
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
    plt.xlabel("Time $t\\,\\omega_\\text{pp}$ (1)")
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
        s=f"t$\\,\\omega_\\text{{pp}}=\\,${time[0]:>5.1f}",
        transform=ax.transAxes
    )
    ax.set(
        xlabel = "Position x$\\,/\\,\\lambda_\\text{D}$ (1)",
        ylabel = "Position y$\\,/\\,\\lambda_\\text{D}$ (1)",
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
        text.set_text(f"t$\\,\\omega_\\text{{pp}}\\,=\\,${time[frame_idx]:>5.1f}")
        return text, quad

    ani = FuncAnimation(fig=fig, func=update, frames=range(len(list(time_steps))))
    if save:
        _saveVideo(
            ani,
            f"e_field_{direction.lower()}_evolution-{label}",
            f"e_field_{direction.lower()}_evolution",
        )
    else:
        return HTML(ani.to_jshtml())

def heatingVsAlphaFlowVelocity(info, save: bool=False):
    files = sorted(V_FLOW_VARIATION_FOLDER.glob("*.h5"))
    n_points: int=10
    plt.style.use(MPLSTYLE_FILE)
    fig, axes = plt.subplots(3, 2, figsize=(FIGURE_FULL_SIZE[0] - 0.3, 6.5), sharex=True)
    i = 0
    for ax_row, species in zip(axes, Species):
        for ax, normalize_temperature in zip(ax_row, [False, True]):
            velocity = np.empty(len(files))
            T_diff = np.empty(len(files))
            T_diff_err = np.empty(len(files))
            for file_idx, filename in enumerate(files):
                velocity[file_idx] = int(filename.stem[-3:])
                with h5py.File(filename) as f:
                    temp = physics.kelvinToElectronVolt(
                        np.mean(f[f"Derived/Temperature/{species.value}"], axis=(1,2))
                    )
                T_diff[file_idx] = np.mean(temp[-n_points:]) - np.mean(temp[:n_points])
                T_diff_err[file_idx] = np.sqrt(
                    np.var(temp[-n_points:]) + np.var(temp[:n_points])
                ) / np.sqrt(n_points)
            if normalize_temperature:
                K_alpha_t0 = (info.alpha.si_mass * (velocity * 1e3) ** 2 / (2 * constants.electron_volt))
                T_diff /= K_alpha_t0
                T_diff_err /= K_alpha_t0
            ax.errorbar(
                velocity, T_diff, yerr=T_diff_err,
                marker="p" if normalize_temperature else "o",
                color="cornflowerblue" if normalize_temperature else "white",
                ls="", markeredgecolor="black", markeredgewidth=1
            )
            ax.text(0.05, 0.93,
                horizontalalignment='left',
                verticalalignment='top',
                s=rf"$\mathbf{{({chr(ord('a')+i)})}}$",
                transform=ax.transAxes
            )
            i += 1
            ax.set(
                xlim=(95, 185),
                xticks=np.arange(100, 185, 20),
            )
            if normalize_temperature:
                ax.ticklabel_format(style='sci', axis='y', scilimits=(-2,2), useMathText=True)
                ax.set_ylabel(f"Heating $\\Delta T_{species.symbol()}\\,/\\,K_\\alpha^{{t=0}}$ (1)")
            else:
                ax.set_ylabel(f"Heating $\\Delta T_{species.symbol()}$ (eV)")
    axes[-1,0].set_xlabel(f"Flow velocity $u_\\alpha^{{t=0}}$ (km$\\,/\\,$s)")
    axes[-1,1].set_xlabel(f"Flow velocity $u_\\alpha^{{t=0}}$ (km$\\,/\\,$s)")
    for ax, sp_name in zip(axes[:,1], ["Electrons", "Protons", "Alpha particles"]):
        ax.text(
            1.05, 0.5,
            horizontalalignment='left',
            verticalalignment='center',
            s=sp_name,
            rotation=90,
            transform=ax.transAxes,
        )
    fig.tight_layout(h_pad=0.2)
    if save:
        _saveFigure("heating_vs_u-alpha", sub_folder="alpha_flow_velocity_variation")

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
            s=f"t$\\,\\omega_\\text{{pp}}=\\,${time[0]:>5.1f}",
            color="white",
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

    tight_bbox = fig.get_tightbbox()
    fig.set_size_inches(tight_bbox.width, tight_bbox.height)
    fig.set_layout_engine("tight", pad=0.01)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    def update(frame_idx):
        quad.set_array(f_v[frame_idx].T)
        text.set_text(f"t$\\,\\omega_\\text{{pp}}\\,=\\,${time[frame_idx]:>5.1f}")
        return (quad, text,)

    frames = list(range(len(list(time_steps))))
    ani = FuncAnimation(fig=fig, func=update, frames=frames)
    if save:
        _saveVideo(
            ani,
            f"px_py_distribution-{species}-{label}",
            f"px_py_distribution/{species}"
        )
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
        _saveFigure(f"px_py_distribution-{species}-t={time}")

def magneticFieldDirectionElectricField(
    info: RunInfo,
    save: bool=False,
):
    plt.style.use(MPLSTYLE_FILE)
    plt.figure(figsize=FIGURE_FULL_SIZE)
    for files, label, marker in zip([
            sorted(V_FLOW_VARIATION_FOLDER.glob("*.h5")),
            sorted((FOLDER_2D / "v_alpha_bulk_variation_Bx").glob("*.h5")),
            sorted((FOLDER_2D / "v_alpha_bulk_variation_By").glob("*.h5")),
            sorted((FOLDER_2D / "v_alpha_bulk_variation_Bz").glob("*.h5")),
            sorted((FOLDER_2D / "v_alpha_bulk_variation_Bx_By").glob("*.h5")),
        ], ["B=0", "B=B_x", "B=B_y", "B=B_z", "B_x=B_y"], MARKERS):
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
        if True:
            K_alpha_t0 = (info.alpha.si_mass * info.alpha.number_density * (velocity * 1e3) ** 2 / (2 * constants.electron_volt))
            W_E_max /= K_alpha_t0
            W_E_max_err /= K_alpha_t0

        plt.errorbar(
            velocity, W_E_max, yerr=W_E_max_err, label=f"${label}$", # color="black",
            marker=marker, markeredgecolor="black", markeredgewidth=1, markersize=10, ls="")
        y_min, y_max = plt.gca().get_ylim()
    plt.fill_between(
        [0.0, info.c_s * 1e-3], y1=y_min, y2=y_max,
        lw=2, color="lightgray", label="$u_{\\alpha} \\leq c_s$"
    )
    plt.xlabel("Flow velocity $u_{\\alpha}^{t=0}$ (km/s)")
    plt.ylabel("$\\max[\\langle W_E\\rangle_\\mathbf{r}]_t\\,/\\,K_\\alpha^{t=0}$  (1)")
    plt.xlim(90, 190)
    plt.ylim(y_min, y_max)
    plt.legend(fancybox=False, edgecolor="black")
    plt.tight_layout()
    if save:
        _saveFigure("max_electric_field_energy-vs-flow-velocity", "magentic_field_direction")

def potentialFromElectricField(E_x, E_y, x, y):
    Nx, Ny = E_x.shape[-2:]
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    # Fourier frequencies
    k_x = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    k_y = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    K_X, K_Y = np.meshgrid(k_x, k_y, indexing='ij')
    K = K_X ** 2 + K_Y ** 2
    K[0, 0] = 1.0  # Avoid division by zero
    # FFT of E field
    E_x_fft = np.fft.fft2(E_x)
    E_y_fft = np.fft.fft2(E_y)
    # Compute phi in Fourier space
    phi_fft = 1j * (K_X * E_x_fft + K_Y * E_y_fft) / K
    phi_fft[0, 0] = 0.0  # Set mean to zero
    # Inverse FFT to get phi
    phi = np.real(np.fft.ifft2(phi_fft))
    return phi

def _pxPyDistSubplot(fig, ax: plt.Axes, info: RunInfo, filename: Path, species: Species, time: float, xlim, ylim, colorbar, regimes):
    f_v0 = _loadPxPyDistribution(info, species, filename, 0, True)[2]
    f_v_max = np.nanmax(f_v0)
    f_v_min = np.nanmin(f_v0[f_v0>0] / f_v_max)

    v_x, v_y, f_v = _loadPxPyDistribution(info, species, filename, time, True)
    dv_x = abs(v_x[1] - v_x[0])
    dv_y = abs(v_y[1] - v_y[0])
    v_x = np.concat([[v_x[0]-dv_x], v_x]) + dv_x / 2
    v_y = np.concat([[v_y[0]-dv_y], v_y]) + dv_y / 2
    f_v /= f_v_max
    f_v[f_v<=f_v_min] = f_v_min
    quad = ax.pcolormesh(v_x, v_y, f_v.T, norm="log", vmin=f_v_min, vmax=1.0, rasterized=True)

    u_alpha = int(filename.stem[-3:])
    ax.plot(
        0 if species == Species.PROTON else u_alpha * 1e3 / info.alpha.v_thermal, 0,
        marker="o", markeredgecolor="black", markeredgewidth=1, color="white")
    with h5py.File(filename) as f:
        sim_time = f['Header/time'][:]
        time_idx = np.argmin(np.abs(sim_time - time))
        Ex = f['Electric Field/ex'][time_idx]
        Ey = f['Electric Field/ey'][time_idx]
        x = f['Grid/grid/X'][:]
        y = f['Grid/grid/Y'][:]
    phi = np.max(np.abs(potentialFromElectricField(Ex, Ey, x, y)))
    with h5py.File(THEORY_U_ALPHA_FILE) as f:
        theory_v = f["u_alpha_bulk"][:] / 1e3
        theory_k = f["k_max"][:]
        theory_omega = f["omega_max"][:]
        theory_theta = f["theta_max"][:]

    v_ph = np.mean((theory_omega / theory_k)[theory_v>100]) / info[species].v_thermal
    v_trap = np.sqrt(2 * info[species].si_charge * phi / info[species].si_mass) / info[species].v_thermal

    theta = theory_theta[np.argmin(np.abs(theory_v - int(u_alpha)))]
    rect_pos = plt.Rectangle(
        xy=(
            (v_ph - v_trap) * np.cos(theta) + np.sin(theta) * (-4 * v_ph),
            (v_ph - v_trap) * np.sin(theta) - np.cos(theta) * (-4 * v_ph)
        ),
        width=100, height=2 * v_trap, angle=-(np.pi/2 - theta) * 180 / np.pi,
        edgecolor="black", zorder=1, facecolor="#c7c7c7", alpha=0.2
    )
    rect_neg = plt.Rectangle(
        xy=(
            +(v_ph + v_trap) * np.cos(theta) + np.sin(theta) * (-4 * v_ph),
            -(v_ph + v_trap) * np.sin(theta) + np.cos(theta) * (-4 * v_ph)
        ),
        width=100, height=2 * v_trap, angle=(np.pi/2 - theta) * 180 / np.pi,
        edgecolor="black", zorder=1, facecolor="#c7c7c7", alpha=0.2
    )
    if regimes:
        delta = 0 if species == Species.PROTON else 4 * v_trap
        alpha = theta
        arrow_fix = 0.24 if species == Species.PROTON else 0.18
        ax.annotate(
            text='', xy=(
                (v_ph - v_trap - arrow_fix) * np.cos(theta) + np.sin(theta) * delta,
                (v_ph - v_trap - arrow_fix) * np.sin(theta) - np.cos(theta) * delta
            ),
            xytext=(
                (v_ph + v_trap + arrow_fix) * np.cos(theta) + np.sin(theta) * delta,
                (v_ph + v_trap + arrow_fix) * np.sin(theta) - np.cos(theta) * delta
            ), arrowprops=dict(arrowstyle='<->', lw=1.2, color="#000000")
        )
        ax.annotate(
            text='', xy=(
                +(v_ph - v_trap - arrow_fix) * np.cos(theta) + np.sin(theta) * delta,
                -(v_ph - v_trap - arrow_fix) * np.sin(theta) + np.cos(theta) * delta
            ),
            xytext=(
                +(v_ph + v_trap + arrow_fix) * np.cos(theta) + np.sin(theta) * delta,
                -(v_ph + v_trap + arrow_fix) * np.sin(theta) + np.cos(theta) * delta
            ), arrowprops=dict(arrowstyle='<->', lw=1.2, color="#000000")
        )
        ax.add_patch(rect_pos)
        ax.add_patch(rect_neg)
    if colorbar:
        fig.colorbar(
            quad, ax=ax, location='top',
            label=f"$\\langle f_{species.symbol()}\\rangle_\\mathbf{{r}}$ (a.u.)",
            fraction=0.15,
            aspect=15,
            pad=0.03,
            use_gridspec=False,
            shrink=0.92 if species == Species.PROTON else 0.83
        )
    ax.set(
        xlim = xlim,
        ylim = ylim,
        xticks=np.arange(xlim[0], xlim[1]+1, 3),
        yticks=np.arange(ylim[0], ylim[1]+1, 3 if ylim[0] % 2 == 0 else 2),
        ylabel=f"Velocity $v_{{{species.symbol()},y}}\\,/\\,v^{{t=0}}_{{\\text{{t}}{species.symbol()}}}$ (1)",
    )
    ax.set_aspect("equal")

def velocitySpaceVsFlowVelocity(info, filename, save: bool=True):
    plt.style.use(MPLSTYLE_FILE)
    fig, axes = plt.subplots(
        2, 2, sharex="col", layout="constrained",
        figsize=(FIGURE_FULL_SIZE[0]-0.5, 5.4),
    )

    i = 0
    for row_idx, (ax_row, time, colorbar) in enumerate(zip(axes, [55.0, 150.0], [True, False])):
        for ax, species, xlim, ylim in zip(ax_row, [Species.PROTON, Species.ALPHA], [(-3, 9), (0, 9)], [(-6, 6), (-5, 5)]):
            _pxPyDistSubplot(fig, ax, info, filename, species, time, xlim, ylim, colorbar, colorbar)
            if row_idx == 1:
                ax.set_xlabel(f"Velocity $v_{{{species.symbol()},x}}\\,/\\,v^{{t=0}}_{{\\text{{t}}{species.symbol()}}}$ (1)")
            ax.text(
                0.03, 0.96,
                s=rf"$\mathbf{{({chr(ord('a')+i)})}}\,\,t\,\omega_\text{{pp}}={time:.0f}$",
                horizontalalignment="left",
                verticalalignment="top",
                color="white",
                transform=ax.transAxes,
            )
            i += 1
    if save:
        _saveFigure(f"velocity_space-flow_velocity_{filename.stem[-3:]}", "alpha_flow_velocity_variation")

def waveNumberVsMagneticField(info: RunInfo, save: bool=False):
    plt.style.use(MPLSTYLE_FILE)
    plt.figure(figsize=(FIGURE_HALF_SIZE[0], 2.5))
    folder_suffixes = ["", "_Bx", "_By", "_Bz", "_Bx_By"]
    field_labels = ["$B$=0", r"$B$=$B_x$", r"$B$=$B_y$", r"$B$=$B_z$", r"$B_x$=$B_y$"]
    for u_label, marker, offset, mean_color, marker_color, ls in zip(
        [100, 140], MARKERS, [-0.1, 0.1],
        ["black", "cornflowerblue"], ["white", "cornflowerblue"],
        ["-", "--"],
    ):
        k_arr = []
        k_err_arr = []
        for folder_id in folder_suffixes:
            for filename in sorted((FOLDER_2D / f"v_alpha_bulk_variation{folder_id}").glob("*.h5")):
                u = int(filename.stem[-3:])
                if u != u_label:
                    continue
                with h5py.File(filename) as f:
                    time = f["Header/time"][1:] * info.omega_pp
                    if f["Grid/x_px/Alphas/X"].ndim > 1:
                        x = f["Grid/x_px/Alphas/X"][0] / info.lambda_D
                        y = f["Grid/y_px/Alphas/Y"][0] / info.lambda_D
                    else:
                        x = f["Grid/x_px/Alphas/X"][:] / info.lambda_D
                        y = f["Grid/y_px/Alphas/Y"][:] / info.lambda_D
                    E_x = f["Electric Field/ex"][1:]
                    E_y = f["Electric Field/ey"][1:]
                res = analysis.fitGrowthRate(time, np.mean(E_x ** 2 + E_y ** 2, axis=(1,2)))
                linear_idx = slice(res[1][-1])
                E_field = E_x[linear_idx]
                k, k_err = analysis.waveVector2D(x, y, E_field)
                k_arr.append(np.linalg.norm(k))
                k_err_arr.append(np.linalg.norm(k_err))
                # omega, omega_err = analysis.estimateFrequency(-3, [0, 0.1], E_field, n_spatial_dims=2)
        k = np.array(k_arr)
        k_err = np.array(k_err_arr)
        # compute chi^2 p-value
        mean = np.sum(k / k_err ** 2) / np.sum(1 / k_err ** 2)
        chi_2 = np.sum((k - mean) ** 2 / k_err ** 2)
        print("Chi2 p-value ", 1 - stats.chi2.cdf(chi_2, df=k.size - 1))
        # plot values
        plt.errorbar(
            np.arange(k.size) + offset, k, yerr=k_err,
            marker=marker, ls="", mfc=marker_color, ecolor=mean_color)
        plt.axhline(mean, color=mean_color, ls=ls)
        plt.errorbar(0, 0, yerr=0.1, marker=marker, ls=ls,
            color=mean_color, mfc=marker_color, ecolor=mean_color, label=rf"{u_label}")

    hatch_width=0.16
    for x in np.arange(6)-0.5:
        plt.gca().add_patch(
            plt.Rectangle(
                (x-hatch_width/2, 0), hatch_width, 2,
                fill=False, hatch="/////", edgecolor="black", zorder=0))
        plt.axvline(x-hatch_width/2, color="white", lw=1, zorder=1)
        plt.axvline(x+hatch_width/2, color="white", lw=1, zorder=1)

    plt.gca().set(
        xlim=(-0.5-hatch_width/2, 4.5+hatch_width/2),
        xticks=np.arange(5)-0.5,
        xticklabels = [''] * 5,
        ylim = (0.52, 0.94),
        ylabel=r"Wave number $k_\text{max}\,\lambda_\text{D}$",
        xlabel=r"B-field configurations"
    )
    for x, l in zip(np.arange(5), field_labels):
        plt.text(x, 0.56, s=l, verticalalignment="top", horizontalalignment="center", fontsize=9)
    plt.tick_params(axis='x', which='minor', length=0)
    plt.tick_params(axis='x', which='major', length=0)
    plt.legend(
        labelspacing=0.4, title=r"Flow velocity $u_\alpha$ (km$\,/\,$s)",
        fontsize=9.2, loc=(0.1,0.74), ncols=2, columnspacing=1, markerscale=0.8)
    if save:
        _saveFigure("wave_number-vs-magnetic_field_direction", "magnetic_fields")

def frequencyVsMagneticField(info: RunInfo, save: bool=False):
    plt.style.use(MPLSTYLE_FILE)
    plt.figure(figsize=(FIGURE_HALF_SIZE[0], 2.5))
    folder_suffixes = ["", "_Bx", "_By", "_Bz", "_Bx_By"]
    field_labels = ["$B$=0", r"$B$=$B_x$", r"$B$=$B_y$", r"$B$=$B_z$", r"$B_x$=$B_y$"]
    for u_label, marker, offset, mean_color, marker_color, ls in zip(
        [100, 140], MARKERS, [-0.1, 0.1],
        ["black", "cornflowerblue"], ["white", "cornflowerblue"],
        ["-", "--"],
    ):
        k_arr = []
        k_err_arr = []
        for folder_id in folder_suffixes:
            for filename in sorted((FOLDER_2D / f"v_alpha_bulk_variation{folder_id}").glob("*.h5")):
                u = int(filename.stem[-3:])
                if u != u_label:
                    continue
                with h5py.File(filename) as f:
                    time = f["Header/time"][1:] * info.omega_pp
                    if f["Grid/x_px/Alphas/X"].ndim > 1:
                        x = f["Grid/x_px/Alphas/X"][0] / info.lambda_D
                        y = f["Grid/y_px/Alphas/Y"][0] / info.lambda_D
                    else:
                        x = f["Grid/x_px/Alphas/X"][:] / info.lambda_D
                        y = f["Grid/y_px/Alphas/Y"][:] / info.lambda_D
                    E_x = f["Electric Field/ex"][1:]
                    E_y = f["Electric Field/ey"][1:]
                res = analysis.fitGrowthRate(time, np.mean(E_x ** 2 + E_y ** 2, axis=(1,2)))
                linear_idx = slice(res[1][-1])
                E_field = E_x[linear_idx]
                omega, omega_err = analysis.estimateFrequency(-3, [0, 0.1], E_field, n_spatial_dims=2)
                k_arr.append(omega)
                k_err_arr.append(omega_err)
        k = np.array(k_arr)
        k_err = np.array(k_err_arr)
        # compute chi^2 p-value
        mean = np.sum(k / k_err ** 2) / np.sum(1 / k_err ** 2)
        chi_2 = np.sum((k - mean) ** 2 / k_err ** 2)
        print("Chi2 p-value ", 1 - stats.chi2.cdf(chi_2, df=k.size - 1))
        # plot values
        plt.errorbar(
            np.arange(k.size) + offset, k, yerr=k_err,
            marker=marker, ls="", mfc=marker_color, ecolor=mean_color)
        plt.axhline(mean, color=mean_color, ls=ls)
        plt.errorbar(0, 0, yerr=0.1, marker=marker, ls=ls,
            color=mean_color, mfc=marker_color, ecolor=mean_color, label=rf"{u_label}")

    hatch_width=0.16
    for x in np.arange(6)-0.5:
        plt.gca().add_patch(
            plt.Rectangle(
                (x-hatch_width/2, 0), hatch_width, 2,
                fill=False, hatch="/////", edgecolor="black", zorder=0))
        plt.axvline(x-hatch_width/2, color="white", lw=1, zorder=1)
        plt.axvline(x+hatch_width/2, color="white", lw=1, zorder=1)

    plt.gca().set(
        xlim=(-0.5-hatch_width/2, 4.5+hatch_width/2),
        xticks=np.arange(5)-0.5,
        xticklabels = [''] * 5,
        ylim = (0.66, 1.07),
        xlabel=r"B-field configurations",
        ylabel=r"Frequency $\omega_\text{max}\,/\,\omega_\text{pp}$",
    )
    for x, l in zip(np.arange(5), field_labels):
        plt.text(x, 0.7, s=l, verticalalignment="top", horizontalalignment="center", fontsize=9)
    plt.tick_params(axis='x', which='minor', length=0)
    plt.tick_params(axis='x', which='major', length=0)
    plt.legend(
        labelspacing=0.4, title=r"Flow velocity $u_\alpha$ (km$\,/\,$s)",
        fontsize=9.2, loc=(0.1,0.74), ncols=2, columnspacing=1, markerscale=0.8)
    if save:
        _saveFigure("frequency-vs-magnetic_field_direction", "magnetic_fields")

def heatingVsMagneticField(info: RunInfo, species: Species, save: bool=False):
    Y_LIM = {
        Species.ELECTRON: (0.7, 4.0),
        Species.PROTON: (0.85, 1.02),
        Species.ALPHA: (0.88, 1.02),
    }
    TEXT_X = {
        Species.ELECTRON: 1.05,
        Species.PROTON: 0.868,
        Species.ALPHA: 0.895,
    }
    LEGEND_LOC = {
        Species.ELECTRON: (0.03, 0.35),
        Species.PROTON:"best",
        Species.ALPHA: (0.1, 0.72),
    }
    plt.style.use(MPLSTYLE_FILE)
    plt.figure(figsize=(FIGURE_HALF_SIZE[0], 2.5))
    folder_suffixes = ["", "_Bx", "_By", "_Bz", "_Bx_By"]
    field_labels = ["$B$=0", r"$B$=$B_x$", r"$B$=$B_y$", r"$B$=$B_z$", r"$B_x$=$B_y$"]
    for u_label, marker, markercolor in zip(
        [100, 140], MARKERS, ["white", "cornflowerblue"]
    ):
        t_diff_arr = []
        for folder_id in folder_suffixes:
            for filename in sorted((FOLDER_2D / f"v_alpha_bulk_variation{folder_id}").glob("*.h5")):
                u = int(filename.stem[-3:])
                if u != u_label:
                    continue
                with h5py.File(filename) as f:
                    time = f["Header/time"][1:] * info.omega_pp
                    temp = np.mean(f[f"Derived/Temperature/{species.value}"], axis=(1,2))
                t_diff = physics.kelvinToElectronVolt(np.mean(temp[-10:]) - np.mean(temp[:10]))
                t_diff_arr.append(t_diff)
        t_diff_arr = np.array(t_diff_arr)
        plt.errorbar(np.arange(4), t_diff_arr[1:] / t_diff_arr[0],
            marker=marker, label=rf"${u_label}$" + (r"$\,$km$\,/\,$s" if species != Species.ALPHA else ""),
            ls="", color=markercolor, mec="black", mew=1.0)
    for x, l in zip(np.arange(4), field_labels[1:]):
        plt.text(x, TEXT_X[species], s=l, verticalalignment="top", horizontalalignment="center", fontsize=9)
    hatch_width=0.16
    for x in np.arange(5)-0.5:
        plt.gca().add_patch(
            plt.Rectangle(
                (x-hatch_width/2, 0), hatch_width, 5,
                fill=False, hatch="/////", edgecolor="black", zorder=0))
        plt.axvline(x-hatch_width/2, color="white", lw=1, zorder=1)
        plt.axvline(x+hatch_width/2, color="white", lw=1, zorder=1)
    plt.legend(title=r"Flow velocity $u_\alpha$" + (r" (km$\,/\,$s)" if species == Species.ALPHA else ""), loc=LEGEND_LOC[species],
               labelspacing=0.4, ncols=2 if species== Species.ALPHA else 1, columnspacing=.5)
    plt.gca().set(
        xlabel=r"B-field configurations",
        ylabel=rf"Heating $\Delta T_{{{species.symbol()},B}}/\Delta T_{{{species.symbol()},B=0}}$",
        ylim=Y_LIM[species],
        xlim=(-0.5-hatch_width/2, 3.5+hatch_width/2),
        xticks=[],
    )
    if species == Species.ALPHA:
        plt.yticks([0.9, 0.95, 1.0])
    plt.tick_params(axis='x', which='minor', length=0)
    plt.tick_params(axis='x', which='major', length=0)
    if save:
        _saveFigure(f"heating_{species.value.lower()}-vs-magnetic_field_direction", "magnetic_fields")

def tempeatureOverTimeVsMagneticField(info: RunInfo, species: Species, save: bool=False):
    # plot 3D temperature vs time for different magnetic fields.
    plt.style.use(MPLSTYLE_FILE)
    plt.figure()

    leg_u100 = []
    leg_u140 = []
    for u_label in [100, 140]:
        for folder_id, label in zip(["", "_Bx", "_By", "_Bz", "_Bx_By"], ["B=0", "B_x>0", "B_y>0", "B_z>0", "B_x=B_y"]):
            for filename in sorted((FOLDER_2D / f"v_alpha_bulk_variation{folder_id}").glob("*.h5")):
                u = int(filename.stem[-3:])
                if u != u_label:
                    continue
                with h5py.File(filename) as f:
                    time = f["Header/time"][:] * info.omega_pp
                    temp = np.mean(f[f"Derived/Temperature/{species.value}"], axis=(1,2))
                line = plt.plot(
                    time, physics.kelvinToElectronVolt(temp),
                    label=f"${label}$"
                )[0]
                (leg_u100 if u_label == 100 else leg_u140).append(line)
    leg1 = plt.legend(handles=leg_u100, loc="upper left", title="$u_\\alpha^{t=0}$=100 km/s",ncols=1 if species == Species.ELECTRON else 2, columnspacing=0.5, labelspacing=0.5, fancybox=False, framealpha=0.4)
    plt.legend(handles=leg_u140, title="$u_\\alpha^{t=0}$=140 km/s", loc="lower right", ncols=3 if species == Species.ELECTRON else 2, columnspacing=0.5, labelspacing=0.5, fancybox=False, framealpha=0.4)
    plt.gca().add_artist(leg1)
    plt.xlabel("Time $t\\,\\omega_{pp}$ (1)")
    plt.ylabel(f"Temperature $T_{species.symbol()}$ (eV)")
    plt.xlim(0, 150.0)
    y_low = 1.5 if species == Species.PROTON else 5
    plt.ylim(bottom=2 if species == Species.PROTON else 10 if species == Species.ALPHA else 99.7)
    plt.tight_layout(pad=0.2)
    if save:
        _saveFigure(f"temperature_{species.value.lower()}-vs-time", "magnetic_fields")

def gammaVsFlowVelocity(info: RunInfo, save: bool=False):
    files = sorted(V_FLOW_VARIATION_FOLDER.glob("*.h5"))
    flow_velocity = np.empty(len(files))
    growth_rate = []
    for file_idx, filename in enumerate(files):
        flow_velocity[file_idx] = int(filename.stem[-3:])
        with h5py.File(filename) as f:
            E_x = f['Electric Field/ex'][1:]
            E_y = f['Electric Field/ey'][1:]
            time = f["Header/time"][1:] * info.omega_pp
        res = analysis.fitGrowthRate(time, np.mean(E_x ** 2 + E_y ** 2, axis=(1,2)), allowed_slope_deviation=0.5)
        growth_rate.append(res[0].slope/2)
    growth_rate = np.array(growth_rate)
    plt.plot(flow_velocity, growth_rate, ls="", marker="o", color="white", markeredgecolor="black", markeredgewidth=1)
    with h5py.File(THEORY_U_ALPHA_FILE) as f:
            theory_v = f["u_alpha_bulk"][:] / 1e3
            theory_gamma = f["gamma_max"][:] / info.omega_pp
    plt.plot(theory_v, theory_gamma)
    if save:
        _saveFigure(f"gamma-vs-flow_velocity", "alpha_flow_velocity_variation")

def heatingVsFlowVelocitySpecies(info: RunInfo, species: Species, save: bool=False):
    files = sorted(V_FLOW_VARIATION_FOLDER.glob("*.h5"))
    n_points: int=10

    plt.style.use(MPLSTYLE_FILE)
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_FULL_SIZE[0] - 0.3, 3), sharex=True)
    for ax, normalize_temperature in zip(axes, [False, True]):
        velocity = np.empty(len(files))
        T_diff = np.empty(len(files))
        T_diff_err = np.empty(len(files))
        for file_idx, filename in enumerate(files):
            velocity[file_idx] = int(filename.stem[-3:])
            with h5py.File(filename) as f:
                temp = physics.kelvinToElectronVolt(
                    np.mean(f[f"Derived/Temperature/{species.value}"], axis=(1,2))
                )
            T_diff[file_idx] = np.mean(temp[-n_points:]) - np.mean(temp[:n_points])
            T_diff_err[file_idx] = np.sqrt(
                np.var(temp[-n_points:]) + np.var(temp[:n_points])
            ) / np.sqrt(n_points)
        if normalize_temperature:
            K_alpha_t0 = (info.alpha.si_mass * (velocity * 1e3) ** 2 / (2 * constants.electron_volt))
            T_diff /= K_alpha_t0
            T_diff_err /= K_alpha_t0
        else:
            T_diff /= info[species].temperature
            T_diff_err /= info[species].temperature
        ax.errorbar(
            velocity, T_diff, yerr=T_diff_err,
            marker="p" if normalize_temperature else "o",
            color="cornflowerblue" if normalize_temperature else "white",
            ls="", markeredgecolor="black", markeredgewidth=1
        )
        ax.set(
            xlim=(95, 185),
            xticks=np.arange(100, 185, 20),
        )
        if normalize_temperature:
            ax.ticklabel_format(style='sci', axis='y', scilimits=(-2,2), useMathText=True)
            ax.set_ylabel(f"Heating $\\Delta T_{species.symbol()}\\,/\\,K_\\alpha^{{t=0}}$ (1)")
        else:
            ax.set_ylabel(f"Heating $\\Delta T_{species.symbol()}\\,/\\,T_{species.symbol()}$ (eV)")
    axes[0].set_xlabel(f"Flow velocity $u_\\alpha^{{t=0}}$ (km$\\,/\\,$s)")
    axes[1].set_xlabel(f"Flow velocity $u_\\alpha^{{t=0}}$ (km$\\,/\\,$s)")
    fig.tight_layout(h_pad=0.2)
    if save:
        _saveFigure(f"heating-vs-flow_velocity-{species}", "alpha_flow_velocity_variation")
