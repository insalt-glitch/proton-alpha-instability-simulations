from pathlib import Path

from cycler import cycler
import colormaps as cmaps
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy import constants, signal

import analysis
from basic import RunInfo, physics, Species
from basic.paths import (
    MPLSTYLE_FILE,
    V_FLOW_VARIATION_FOLDER
)
from .general import generalSaveFigure, plotEnergyEFieldOverTime

def _saveFigure(fig_name: str, sub_folder: str|None = None):
    if sub_folder is None:
        sub_folder = "simulation-2D"
    else:
        sub_folder = f"simulation-2D/{sub_folder}"
    generalSaveFigure(fig_name, sub_folder)

def maxEnergyVsAlphaFlowSpeed(info: RunInfo, save: bool=False):
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

    plt.style.use(MPLSTYLE_FILE)
    plt.figure()
    plt.fill_between([0.0, info.c_s * 1e-3], y1=2, y2=8, lw=2, color="lightgray", label="$u_{\\alpha} \\leq c_s$")
    plt.errorbar(
        velocity, W_E_max * 1e-6, yerr=1e-6 * W_E_max_err, color="black",
        marker="p", markeredgecolor="black", markeredgewidth=1, markersize=10, ls="")
    plt.xlabel("Initial flow velocity $u_{\\alpha}^{t=0}$ (km/s)")
    plt.ylabel("$\\max[\\langle W_E\\rangle_\\mathbf{r}]_t$  (MeV$\\,/\\,$m$^3$)")
    plt.xlim(90, 190)
    plt.ylim(2, 7.5)
    plt.legend(fancybox=False, edgecolor="black")
    plt.tight_layout()
    if save:
        _saveFigure("max_energy-vs-alpha_flow_velocity", "alpha_flow_velocity_variation")

def maxEnergyNormalizedVsAlphaFlowSpeed(info: RunInfo, save: bool=False):
    velocity = []
    W_E_max = []
    W_E_max_err = []
    for file in sorted(V_FLOW_VARIATION_FOLDER.glob("*.h5")):
        velocity.append(int(file.stem[-3:]))
        with h5py.File(file) as f:
            E_x = f['Electric Field/ex'][:]
            E_y = f['Electric Field/ey'][:]
        W_E = np.mean(E_x ** 2 + E_y ** 2, axis=(1,2)) * (constants.epsilon_0 / 2) / constants.electron_volt
        max_idx = np.argmax(W_E)
        max_range = W_E[max_idx-5:max_idx+5]
        W_E_max.append(np.mean(max_range))
        W_E_max_err.append(np.std(max_range))

    velocity = np.array(velocity) * 1e3
    W_E_alpha_t0 = (info.alpha.si_mass * info.alpha.number_density * velocity ** 2 / (2 * constants.electron_volt))
    W_E_max = np.array(W_E_max) / W_E_alpha_t0
    W_E_max_err = np.array(W_E_max_err) / W_E_alpha_t0
    plt.style.use(MPLSTYLE_FILE)
    plt.figure()
    plt.errorbar(
        velocity * 1e-3, W_E_max, yerr=W_E_max_err, color="black",
        marker="p", markeredgecolor="black", markeredgewidth=1, markersize=10, ls="")
    plt.xlabel("Initial flow velocity $u_{\\alpha}^{t=0}$ (km/s)")
    plt.ylabel("$\\max[\\langle W_E\\rangle_\\mathbf{r}]_t/\\langle K_{\\alpha}^{t=0}\\rangle_\\mathbf{r}$ (1)")
    plt.xlim(90, 190)
    if save:
        _saveFigure("max_energy_normalized-vs-alpha_flow_velocity", "alpha_flow_velocity_variation")

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

def electricField2DSnapshot(data_file: Path, info: RunInfo, time: float|int, save: bool=False):
    with h5py.File(data_file) as f:
        if isinstance(time, float):
            time_step = np.argmin(np.abs(f["Header/time"][:] * info.omega_pp - time))
        else:
            assert abs(time) < f["Header/time"].size, "Time out of range"
            time_step = time
        x = f["Grid/grid/X"][:] / info.lambda_D
        y = f["Grid/grid/Y"][:] / info.lambda_D
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

def energyEFieldOverTime(data_file: Path, info: RunInfo, save: bool=False):
    with h5py.File(data_file) as f:
        E_x = f['Electric Field/ex'][:]
        E_y = f['Electric Field/ey'][:]
        time = f["Header/time"][:] * info.omega_pp
    energy = np.mean(E_y ** 2, axis=(1,2)) * (constants.epsilon_0 / 2) / constants.electron_volt
    plt.style.use(MPLSTYLE_FILE) # E_x ** 2 +
    plt.figure()
    plotEnergyEFieldOverTime(time, energy, False)
    plt.ylim(1e4, 4e6)
    if save:
        _saveFigure(f"electric_field-vs-time", "alpha_flow_velocity_variation")

def strengthBFieldOverTime(data_file: Path, info: RunInfo, save: bool=False):
    with h5py.File(data_file) as f:
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

def psdBField(data_file: Path, info: RunInfo, save: bool=False):
    with h5py.File(data_file) as f:
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

def energyBField(data_file: Path, info: RunInfo, save: bool=False):
    with h5py.File(data_file) as f:
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
    for data_file in sorted(V_FLOW_VARIATION_FOLDER.glob("*.h5")):
        v = int(data_file.stem[-3:])
        with h5py.File(data_file) as f:
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
    info: RunInfo, species: Species, n_points: int=20, save: bool=False
):
    files = sorted(V_FLOW_VARIATION_FOLDER.glob("*.h5"))
    velocity = np.empty(len(files))
    T_diff = np.empty(len(files))
    T_diff_err = np.empty(len(files))
    for file_idx, data_file in enumerate(files):
        velocity[file_idx] = int(data_file.stem[-3:])
        with h5py.File(data_file) as f:
            time = f["Header/time"][:] * info.omega_pp
            temp = physics.kelvinToElectronVolt(
                np.mean(f[f"Derived/Temperature/{species.value}"], axis=(1,2))
            )
        T_diff[file_idx] = np.mean(temp[-n_points:]) - np.mean(temp[:n_points])
        T_diff_err[file_idx] = np.sqrt(
            np.var(temp[-n_points:]) + np.var(temp[n_points:])
        ) / np.sqrt(n_points)
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
    plt.ylabel(f"Temperature $\\Delta T_{species.symbol()}$ (eV)")
    plt.legend(loc="lower right")
    if save:
        _saveFigure(f"temp_3D_{species.value.lower()}-vs-alpha_flow_velocity", "alpha_flow_velocity_variation")

def psdFlowVelocity(info: RunInfo, species: Species, space_dir: str, mom_dir: str, save: bool=False):
    assert space_dir in ["x", "y"], "Unknown e-field component"
    assert mom_dir in ["x", "y"], "Unknown e-field component"
    plt.style.use(MPLSTYLE_FILE)
    plt.figure()
    for data_file in sorted(V_FLOW_VARIATION_FOLDER.glob("*.h5")):
        u_alpha = int(data_file.stem[-3:])
        with h5py.File(data_file) as f:
            time = f["Header/time"][:] * info.omega_pp
            x_grid = f[f"Grid/{space_dir}_p{mom_dir}/{species.value}/{space_dir.upper()}"][:]
            px_grid = f[f"Grid/{space_dir}_p{mom_dir}/{species.value}/P{mom_dir}"][:]
            dist_x_px = f[f'dist_fn/{space_dir}_p{mom_dir}/{species.value}'][:]

        u_x = analysis.flowVelocity1D(x_grid, px_grid, dist_x_px, info[species])
        f, Pxx = signal.periodogram(x=np.mean(u_x, axis=1), fs=1 / (0.1))
        plt.plot(f[1:], Pxx[1:], label=f"$u_\\alpha^{{t=0}}$={u_alpha}")
        if species == Species.ELECTRON:
            break
    plt.xlabel("Frequency $f\\,/\\,\\omega_{pp}$ (1)")
    plt.ylabel(f"PSD$[\\langle u_{{e,{mom_dir}}}\\rangle_{space_dir}]$ (km$^2\\,/\\,$s$^2$)")
    plt.yscale("log")
    plt.xscale("log")
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
    for data_file in sorted(V_FLOW_VARIATION_FOLDER.glob("*.h5")):
        u_alpha = int(data_file.stem[-3:])
        with h5py.File(data_file) as f:
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
