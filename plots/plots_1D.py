from pathlib import Path

import colormaps as cmaps
import matplotlib.legend_handler
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy import constants, signal, io
import h5py

import analysis
import theory
from basic import physics, Distribution
from .general import generalSaveFigure, plotEnergyEFieldOverTime, _loadSpaceMomDistribution

from basic.paths import (
    RESULTS_FOLDER,
    FOLDER_1D,
    PARTICLE_VARIATION_FOLDER,
    MPLSTYLE_FILE,
)
from plots.settings import FIGURE_HALF_SIZE, FIGURE_FULL_SIZE
from basic import RunInfo, Species

def _saveFigure(fig_name: str, sub_folder: str|None = None):
    if sub_folder is None:
        sub_folder = "simulation-1D"
    else:
        sub_folder = f"simulation-1D/{sub_folder}"
    generalSaveFigure(fig_name, sub_folder)

def electricFieldOverSpaceAndTime(
    filename: Path,
    info: RunInfo,
    save: bool=False
):
    with h5py.File(filename) as f:
        time = f["Header/time"][:] * info.omega_pp
        grid_edges = f["Grid/grid"][:]
        ex = f["Electric Field/ex"][:]

    grid_edges = grid_edges[0] / info.lambda_D

    plt.style.use(MPLSTYLE_FILE)
    plt.figure(figsize=(FIGURE_FULL_SIZE[0], 3.5))
    plt.pcolormesh(time, grid_edges, ex[:-1].T, cmap="bwr", rasterized=True,
                   vmin=-np.max(np.abs(ex)), vmax=np.max(np.abs(ex)))
    plt.colorbar(label="Electric field $E_x$ (V/m)")
    plt.xlabel("Time $t\\,\\omega_\\text{pp}$ (1)")
    plt.ylabel("Position $x\\,/\\,\\lambda_\\text{D}$ (1)")
    plt.yticks(np.arange(5) * 32)
    plt.ylim(0, np.max(grid_edges))
    plt.xlim(0, 150)
    plt.xticks(np.linspace(0, 150, 6))
    plt.tight_layout()
    if save:
        _saveFigure("e_field-vs-time-vs-space")

def avgTemperature3DOverTime(
    filename: Path,
    info: RunInfo,
    save: bool=False
):
    with h5py.File(filename) as f:
        time = f["Header/time"][:] * info.omega_pp
        T_electron, T_proton, T_alpha = [
            physics.kelvinToElectronVolt(
                np.mean(f[f"Derived/Temperature/{species.value}"][:], axis=1)
            ) for species in Species
        ]
    plt.style.use(MPLSTYLE_FILE)
    plt.figure()
    plt.plot(time, T_electron, ls="--", label="$T_e$")
    plt.plot(time, T_proton, ls=":", label="$T_p$")
    plt.plot(time, T_alpha, ls="solid", label="$T_\\alpha$")
    plt.xlabel("Time $t\\,\\omega_\\text{pp}$ (1)")
    plt.ylabel("Temperature (eV)")
    plt.legend(loc="center left", fontsize=14)
    plt.xlim(time[0], time[-1])
    if save:
        _saveFigure("avg_temp3d-vs-time")

def avgTemperatureXOverTime(
    filename: Path,
    info: RunInfo,
    species: Species,
    save: bool=False
):
    with h5py.File(filename) as f:
        time = f["Header/time"][:]
        dist = np.mean(f[f"/dist_fn/x_px/{species.value}"], axis=1)
        x_grid = f[f"Grid/x_px/{species.value}/X"][:]
        px_grid = f[f"Grid/x_px/{species.value}/Px"][:]
        temp_3d = physics.kelvinToElectronVolt(np.mean(f[f'Derived/Temperature/{species.value}'][:], axis=1))

    time *= info.omega_pp
    temperature = analysis.temperature1D(x_grid, px_grid, dist, info[species])

    plt.style.use(MPLSTYLE_FILE)
    if species == Species.ELECTRON:
        plt.figure(figsize=(FIGURE_HALF_SIZE[0]-0.26, 2.5))
    else:
        plt.figure(figsize=(FIGURE_HALF_SIZE[0], 2.5))
    plt.plot(time, temperature, label=f"$T_{{{species.symbol()},x}}$")
    plt.plot(time, temp_3d, label=f"$T_{{{species.symbol()},\\text{{3D}}}}$", color="#aaaaaa")
    plt.xlabel("Time $t\\,\\omega_\\text{pp}$ (1)")
    plt.ylabel(f"Temperature $T_{{{species.symbol()}}}$ (eV)")
    plt.xlim(0, 150)
    plt.xticks(np.linspace(0, 150, 6))
    plt.legend(labelspacing=0.4, loc="best" if species != Species.ELECTRON else (0.02,0.72))
    if species == Species.ELECTRON:
        plt.yticks([100, 100.5, 101])

    if save:
        _saveFigure(f"avg_temp_x-vs-time_{species.value}")
    else:
        plt.title(species)

def velocityDistributionOverTime(
    filename: Path,
    info: RunInfo,
    species: Species,
    save: bool=False
):
    with h5py.File(filename) as f:
        time = f["Header/time"][:] * info.omega_pp
        x_grid = f[f"Grid/x_px/{species.value}/X"][:]
        px_grid = f[f"Grid/x_px/{species.value}/Px"][:]
        dist = np.mean(f[f"/dist_fn/x_px/{species.value}"][:], axis=1)

    v, f_v = analysis.normalizeDistributionXPx1D(
        x_grid, px_grid, dist, info[species]
    )
    relative_v = v / info[species].v_thermal
    # f_v[f_v<=0] = np.min(f_v[f_v>0])
    if relative_v.ndim > 1:
        time = np.tile(time, (px_grid.shape[1], 1)).T
        f_v = f_v.T

    plt.style.use(MPLSTYLE_FILE)
    fig = plt.figure(figsize=(FIGURE_FULL_SIZE[0],3.5))
    quad = plt.pcolormesh(time, relative_v, f_v.T, norm="log", rasterized=True)
    ax = plt.gca()
    plt.xlabel("Time $t\\,\\omega_\\text{pp}$ (1)")
    plt.ylabel(f"Velocity $v_{species.symbol()}\\,/\\,v^{{t=0}}_{{\\text{{th}},{species.symbol()}}}$ (1)")
    plt.xlim(0,150)
    plt.xticks(np.linspace(0, 150, 6))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cax = plt.colorbar(label=f"$\\langle f_{species.symbol()}\\rangle_x$ (s$\\,/\\,$m$^2$)", cax=cax)
    ax.set_facecolor(cax.cmap.get_under())
    plt.tight_layout()
    if save:
        _saveFigure(f"velocity_dist-vs-time_{species.value}")


def velocityDistributionOverTimeCombined(
    filename: Path,
    info: RunInfo,
    times: list[float],
    v_lim_arr: list[tuple[float|None,float|None]],
    v_tick_arr: list[list[float]],
    save: bool=False,
):
    t_list = []
    v_list = []
    f_v_list = []
    min_f = np.inf
    for species in Species:
        with h5py.File(PARTICLE_VARIATION_FOLDER / "particles_8192/rep_0.h5") as f:
            time = f["Header/time"][:] * info.omega_pp
            x_grid = f[f"Grid/x_px/{species.value}/X"][:]
            px_grid = f[f"Grid/x_px/{species.value}/Px"][:]
            dist = np.mean(f[f"/dist_fn/x_px/{species.value}"][:], axis=1)

        v, f_v = analysis.normalizeDistributionXPx1D(
            x_grid, px_grid, dist, info[species]
        )
        relative_v = v / info[species].v_thermal
        if v.ndim > 1:
            time = np.tile(time, (v.shape[1], 1)).T
        f_v = f_v / np.max(f_v)
        t_list.append(time)
        v_list.append(relative_v)
        f_v_list.append(f_v)
        min_f = min(np.min(f_v[f_v>0]), min_f)

    plt.style.use(MPLSTYLE_FILE)
    fig, axes = plt.subplots(
        2, 3, figsize=(FIGURE_FULL_SIZE[0]+0.5, 7),
        sharey="row", sharex="col",
        height_ratios=[0.3, 1],
        width_ratios=[1,1,1.1]
    )
    axes: list[list[plt.Axes]] = axes
    for ax_idx, (ax, species, v_lim, v_ticks, time, v, f_v) in enumerate(zip(
        axes[1], Species, v_lim_arr, v_tick_arr, t_list, v_list, f_v_list,
    )):
        quad = ax.pcolormesh(
            v, time, f_v, norm="log",
            rasterized=True, cmap=plt.cm.get_cmap("viridis"),
            vmin=min_f, vmax=1.0,
        )
        ax.text(
            0.95, 0.98, s=f"$\\mathbf{{({chr(ord('d')+ax_idx)})}}$",
            horizontalalignment='right',
            verticalalignment='top',
            color="white",
            transform=ax.transAxes,
        )
        ax.set(
            facecolor=plt.cm.get_cmap("viridis").get_under(),
            xlabel = f"$v_{species.symbol()}\\,/\\,v^{{t=0}}_{{\\text{{th}},{species.symbol()}}}$ (1)",
            xlim=v_lim,
            xticks=v_ticks
        )
    divider = make_axes_locatable(axes[0,2])
    empty_ax: plt.Axes = divider.append_axes("right", size="5%", pad=0.1)
    empty_ax.set_axis_off()
    divider = make_axes_locatable(axes[1,2])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(
        quad, cax=cax,
        label=f"Distribution $\\langle f_{species.symbol()}\\rangle_x$ (a.u.)")
    axes[1,0].set(
        ylim = (0,150),
        yticks = np.linspace(0, 140, 8),
        ylabel="Time $t\\,\\omega_\\text{pp}$ (1)",
    )

    for ax_idx, (species, v_lim, v_ticks, ax) in enumerate(zip(
        Species, v_lim_arr, v_tick_arr, axes[0]
    )):
        max_f = - np.inf
        for t in times:
            v, f_v = _loadSpaceMomDistribution(info, species, filename, Distribution.X_PX, t, True)
            f_v = np.mean(f_v, axis=0)
            max_f = max(np.max(f_v), max_f)
            ax.plot(v, f_v / max_f, label=f"$t\\,\\omega_\\text{{pp}}={int(t)}$")
        if ax_idx == 1:
            ax.legend(labelspacing=0.2, loc=(0.45, 0.6), fontsize=10, handletextpad=0.3, handlelength=1.4)
            ax.set_zorder(10)
        ax.text(
            0.95, 0.95 if ax_idx != 1 else 0.5, s=f"$\\mathbf{{({chr(ord('a')+ax_idx)})}}$",
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes,
        )
    axes[0,0].set(
        ylabel=f"$\\langle f_{species.symbol()}\\rangle_x$ (a.u.)",
        yscale="log",
        ylim=(1e-3, None),
    )

    plt.tight_layout(w_pad=-1.8, h_pad=0.04)
    if save:
        _saveFigure(f"velocity_dist-vs-time-all-species")

def energyEFieldOverTime(
    filename: Path,
    info: RunInfo,
    show_fit_details: bool=True,
    save: bool=False
):
    with h5py.File(filename) as f:
        time = f["Header/time"][:] * info.omega_pp
        energy = f["Electric Field/ex"][:] ** 2
    energy = np.mean(
        energy,
        axis=tuple(range(1, energy.ndim))
    ) * (constants.epsilon_0 / 2) / (constants.electron_volt)

    plt.style.use(MPLSTYLE_FILE)
    plt.figure(figsize=FIGURE_FULL_SIZE)
    plotEnergyEFieldOverTime(time, energy, show_fit_details=show_fit_details)
    plt.ylim(1e4, 1e7)
    if save:
        _saveFigure("e_field_energy-vs-time")

def particleVariationTemperature3D(
    species: Species,
    save: bool=False
):
    _, (temperatures,), folders = analysis.readFromVariation(
        folder=PARTICLE_VARIATION_FOLDER,
        dataset_names=[f"Derived/Temperature/{species.value}"],
        processElement=lambda x: physics.kelvinToElectronVolt(np.mean(x, axis=1)),
        recursive=True
    )
    with h5py.File(FOLDER_1D / "proton-alpha-instability-1D.h5") as f:
        p8192_temperature = physics.kelvinToElectronVolt(np.mean(
            f[f"Derived/Temperature/{species.value}"][:temperatures.shape[-1]],
            axis=1
        ))
    # extract particle numbers
    particle_numbers = np.array([int(pfs[0].stem[-4:]) for pfs in folders])
    # compute quantities of interest
    temperatures = np.mean(temperatures[:,:,-20:], axis=-1)
    mean_T = np.mean(temperatures, axis=1)
    std_T = np.std(temperatures, axis=1)
    p8192_T = np.mean(p8192_temperature[-20:])

    plt.style.use(MPLSTYLE_FILE)
    plt.figure()
    plt.errorbar(particle_numbers, mean_T, yerr=std_T,
        ls="", color="white", ecolor="black",
        marker="o", markersize=10, markeredgecolor="black", markeredgewidth=1)
    plt.plot(8192, p8192_T,
        ls="", color="white",
        marker="p", markersize=10, markeredgecolor="black", markeredgewidth=1)
    plt.xscale("log", base=2)
    plt.xlabel("Simulated particles $N_\\text{sim}\\,/\\,N_\\text{cell}$ (1)")
    plt.ylabel("Temperature $T_{final}$ (eV)")
    plt.xlim(2 ** 4, 2 ** 14)
    if save:
        _saveFigure(f"temperature_3D-vs-num_particles_{species.value}", "particles_per_cell/temperature")
    else:
        plt.title(species)

def particleVariationEnergyVsTime(
    info: RunInfo,
    save: bool=False
):
    particle_numbers = []
    time, (energies,), folders = analysis.readFromVariation(
        folder=PARTICLE_VARIATION_FOLDER,
        dataset_names=["/Electric Field/ex"],
        processElement=lambda x: np.mean(np.array(x) ** 2, axis=1),
        recursive=True
    )
    with h5py.File(FOLDER_1D / "proton-alpha-instability-1D.h5") as f:
        p8192_time = f["Header/time"][:]
        p8192_energy = np.mean(f["Electric Field/ex"][:] ** 2, axis=1)

    # fix units of time and energy
    time *= info.omega_pp
    energies *= constants.epsilon_0 / (2.0 * constants.electron_volt)
    p8192_time *= info.omega_pp
    p8192_energy *= constants.epsilon_0 / (2.0 * constants.electron_volt)
    # apply running mean
    energies = np.cumsum(np.mean(energies, axis=1), axis=-1)
    energies = (energies[:,10:] - energies[:,:-10]) / 10
    p8192_energy = np.cumsum(p8192_energy)
    p8192_energy = (p8192_energy[10:] - p8192_energy[:-10]) / 10
    # extract particle numbers
    particle_numbers = np.array([int(pfs[0].stem[-4:]) for pfs in folders])
    plt.style.use(MPLSTYLE_FILE)
    plt.figure()
    for num_p, W_E in zip(particle_numbers, energies):
        if num_p < 128:
            continue
        plt.plot(time[5:-5], W_E, label=num_p)
    plt.plot(p8192_time[5:-5], p8192_energy, label=8192)
    plt.yscale("log")
    plt.xlabel("Time $t\\,\\omega_\\text{pp}$ (1)")
    plt.ylabel("Energy $W_E$ (eV$\\,/\\,$m$^3$)")
    plt.legend(title="Simulated particles $N_\\text{sim}\\,/\\,N_c$", ncols=2)
    plt.xlim(time[0], time[-1])
    plt.xticks(np.linspace(0, 150, 6))
    if save:
        _saveFigure("avg_e_field_energy-vs-time-vs-num_particles", "particles_per_cell")

def particleVariationTemperatureXDiff(
    info: RunInfo,
    species: Species,
    save: bool=False
):
    time, (dist,), folders = analysis.readFromVariation(
        folder = PARTICLE_VARIATION_FOLDER,
        dataset_names=[f"dist_fn/x_px/{species.value}"],
        processElement=lambda x: np.mean(x, axis=1),
        recursive=True
    )
    with h5py.File(PARTICLE_VARIATION_FOLDER / "particles_0032/rep_0.h5") as f:
        x_grid = f[f"Grid/x_px/{species.value}/X"][:]
        px_grid = f[f"Grid/x_px/{species.value}/Px"][:]

    with h5py.File(FOLDER_1D / "proton-alpha-instability-1D.h5") as f:
        p8192_time = f["Header/time"][:time.size]
        p8192_dist = np.mean(f[f"/dist_fn/x_px/{species.value}"][:time.size], axis=1)
        p8192_x_grid = f[f"Grid/x_px/{species.value}/X"][:]
        p8192_px_grid = f[f"Grid/x_px/{species.value}/Px"][:]

    time *= info.omega_pp
    p8192_time *= info.omega_pp
    # extract particle numbers
    particle_numbers = np.array([int(pfs[0].stem[-4:]) for pfs in folders])

    temperature = analysis.temperature1D(
        x_grid, px_grid, dist, info[species]
    )
    T_init = np.mean(temperature[:,:,:10], axis=-1)
    T_final = np.mean(temperature[:,:,-10:], axis=-1)
    T_diff = T_final - T_init
    mean_T_diff = np.mean(T_diff, axis=-1)
    std_T_diff = np.std(T_diff, axis=-1)

    p8192_temperature = analysis.temperature1D(
        p8192_x_grid, p8192_px_grid, p8192_dist, info[species]
    )
    p8192_T_diff = np.mean(p8192_temperature[-10:]) - np.mean(p8192_temperature[:10])

    plt.style.use(MPLSTYLE_FILE)
    plt.figure(figsize=(FIGURE_HALF_SIZE[0],2.5))
    l_vary = plt.errorbar(particle_numbers, mean_T_diff, yerr=std_T_diff,
                ls="", lw=1.5, color="white", ecolor="black",
                marker="o", markersize=10, markeredgecolor="black", markeredgewidth=1.5)
    l_8192 = plt.plot(8192, p8192_T_diff,
        ls="", lw=1.5, color="white",
        marker="p", markersize=10, markeredgecolor="black", markeredgewidth=1.5)[-1]
    plt.xscale("log", base=2)
    plt.xlabel("Super particles $N_\\text{sim}\\,/\\,N_\\text{cell}$ (1)")
    plt.ylabel(f"Temperature $\\Delta T_{{{species.symbol()},x}}$ (eV)")
    plt.legend([(l_vary, l_8192)],
               ["Simulation"],
               loc="lower right" if mean_T_diff[-1] > mean_T_diff[0] else "upper right", markerscale=0.7,
               handler_map={tuple: matplotlib.legend_handler.HandlerTuple(ndivide=None)})
    if save:
        _saveFigure(f"temperature_diff-vs-num_particles_{species.value}", "particles_per_cell/temperature")
    else:
        plt.title(species)


def particleVariationTemperatureXVsTime(
    info: RunInfo,
    species: Species,
    save: bool=False
):
    expected_diffs = [102 - 100, 14.5 - 3, 50 - 12]
    time, (dist,), folders = analysis.readFromVariation(
        folder = PARTICLE_VARIATION_FOLDER,
        dataset_names=[f"/dist_fn/x_px/{species.value}"],
        processElement=lambda x: np.mean(x, axis=1),
        recursive=True
    )
    with h5py.File(PARTICLE_VARIATION_FOLDER / "particles_0032/rep_0.h5") as f:
        x_grid = f[f"Grid/x_px/{species.value}/X"][:]
        px_grid = f[f"Grid/x_px/{species.value}/Px"][:]

    with h5py.File(FOLDER_1D / "proton-alpha-instability-1D.h5") as f:
        p8192_time = f["Header/time"][:time.size]
        p8192_dist = np.mean(f[f"/dist_fn/x_px/{species.value}"][:time.size], axis=1)
        p8192_x_grid = f[f"Grid/x_px/{species.value}/X"][:]
        p8192_px_grid = f[f"Grid/x_px/{species.value}/Px"][:]

    time *= info.omega_pp
    p8192_time *= info.omega_pp
    # extract particle numbers
    particle_numbers = np.array([int(pfs[0].stem[-4:]) for pfs in folders])

    temperature = analysis.temperature1D(x_grid, px_grid, dist, info[species])
    p8192_temperature = analysis.temperature1D(
        p8192_x_grid, p8192_px_grid, p8192_dist, info[species]
    )

    plt.style.use(MPLSTYLE_FILE)
    plt.figure()
    plt.plot(time, np.mean(temperature[2:], axis=1).T, label=particle_numbers[2:])
    plt.plot(p8192_time, p8192_temperature, label=8192)
    plt.xlabel("Time $t\\,\\omega_\\text{pp}$ (1)")
    plt.ylabel("Temperature $T_x$ (eV)")
    plt.xlim(time[0], time[-1])
    plt.legend(title="Simulated particles $N_\\text{sim}\\,/\\,N_c$ (1)", ncols=2)

    if save:
        _saveFigure(f"temperature_x-vs-time-vs-num_particles_{species.value}", "particles_per_cell/temperature")
    else:
        plt.title(species)

def particleVariationWavenumber(
    info: RunInfo,
    save: bool=False
):
    time, (E_fields,), folders = analysis.readFromVariation(
        folder=PARTICLE_VARIATION_FOLDER,
        dataset_names=["/Electric Field/ex"],
        recursive=True
    )
    with h5py.File(PARTICLE_VARIATION_FOLDER / "particles_0032/rep_0.h5") as f:
        grid = np.squeeze(f["/Grid/grid"])

    with h5py.File(FOLDER_1D / "proton-alpha-instability-1D.h5") as f:
        p8192_time = f["Header/time"][:time.size]
        p8192_E_field = f["Electric Field/ex"][:time.size]
        p8192_grid = np.squeeze(f["Grid/grid"])

    # fix units of time and energy
    time *= info.omega_pp
    grid /= info.lambda_D
    p8192_time *= info.omega_pp
    p8192_grid /= info.lambda_D
    # extract particle numbers
    particle_numbers = np.array([int(pfs[0].stem[-4:]) for pfs in folders])
    k, k_err = analysis.estimateFrequency(-1, grid, E_fields[...,:350,:], peak_cutoff=0.9)
    p8192_k, p8192_k_err = analysis.estimateFrequency(
        -1, p8192_grid, p8192_E_field[...,:350,:], peak_cutoff=0.9
    )

    mean_k = np.mean(k, axis=1)
    mean_k_err = np.full_like(mean_k, k_err) / np.sqrt(k.shape[1])
    plt.style.use(MPLSTYLE_FILE)
    plt.figure(figsize=FIGURE_HALF_SIZE)
    l_vary = plt.errorbar(particle_numbers[2:], mean_k[2:], yerr=mean_k_err[2:],
        ls="", lw=1.5, marker="o", color="white", ecolor="black",
        markeredgecolor="black", markeredgewidth=1.5)
    l_8192 = plt.errorbar(8192, p8192_k, yerr=p8192_k_err,
        ls="", lw=1.5, color="white", ecolor="black",
        marker="p", markeredgecolor="black", markeredgewidth=1.5)
    l_theory = plt.axhline(theory.waveNumber(1e-1, info), color="black", ls=":")
    plt.ylim(0.35, 0.7)
    y_min, y_max = plt.gca().get_ylim()
    r_fail = plt.fill_between(
        [0, 2 ** 6.5], y1=y_min, y2=y_max,
        color="red", alpha=0.7
    )
    plt.xscale("log", base=2)
    plt.xlim(0.5 * particle_numbers[0], 2 ** 14)
    plt.xticks(2 ** np.linspace(4, 14, 6))
    plt.ylim(y_min, y_max)
    plt.yticks(np.linspace(0.4, 0.7, num=4))
    plt.xlabel("Super particles $N_\\text{sim}\\,/\\,N_\\text{cell}$ (1)")
    plt.ylabel("Wave number $k_\\text{max}\\,\\lambda_\\text{D}$ (1)")
    plt.legend([l_theory, (l_vary, l_8192), r_fail],
               ["Theory", "Simulation", "No wave"],
               loc=(0.44, 0.05), markerscale=0.7, labelspacing=0.4,
               handler_map={tuple: matplotlib.legend_handler.HandlerTuple(ndivide=None)})
    if save:
        _saveFigure("wavenumber-vs-num_particles", "particles_per_cell")

def particleVariationFrequency(
    info: RunInfo,
    save: bool=False
):
    time, (E_fields,), folders = analysis.readFromVariation(
        folder=PARTICLE_VARIATION_FOLDER,
        dataset_names=["/Electric Field/ex"],
        recursive=True
    )
    with h5py.File(FOLDER_1D / "proton-alpha-instability-1D.h5") as f:
        p8192_time = f["Header/time"][:]
        p8192_E_field = f["Electric Field/ex"][:]

    # fix units of time and energy
    time *= info.omega_pp
    p8192_time *= info.omega_pp
    # extract particle numbers
    particle_numbers = np.array([int(pfs[0].stem[-4:]) for pfs in folders])

    omega, omega_err = analysis.estimateFrequency(-2, time, E_fields[...,:350,:])
    p8192_omega, p8192_omega_err = analysis.estimateFrequency(-2, p8192_time, p8192_E_field[...,:350,:])
    mean_omega = np.mean(omega, axis=1)
    mean_omega_err = np.full_like(mean_omega, omega_err / np.sqrt(omega.shape[1]))
    # mean_omega_err[mean_omega_err < omega_err] = omega_err
    colors = cmaps.devon.discrete(3).colors
    plt.style.use(MPLSTYLE_FILE)
    plt.figure(figsize=FIGURE_HALF_SIZE)
    l_vary = plt.errorbar(particle_numbers[2:], mean_omega[2:], yerr=mean_omega_err[2:],
        ls="", marker="o", color=colors[2], lw=1.5,
        markeredgecolor="black", ecolor="black", markeredgewidth=1.5)
    l_8192 = plt.errorbar(2 ** 13, p8192_omega, yerr=p8192_omega_err,
        ls="", marker="p", color=colors[2], lw=1.5,
        markeredgecolor="black", ecolor="black", markeredgewidth=1.5)

    l_theory = plt.axhline(theory.waveFrequency(1e-1), color="black", ls=":")
    plt.ylim(0.4, 0.8)
    y_min, y_max = plt.gca().get_ylim()
    r_fail = plt.fill_between(
        [0, 2 ** 6.5],
        y1=y_min, y2=y_max,
        color="red", alpha=0.7
    )
    plt.xscale("log", base=2)
    plt.xlim(0.5 * particle_numbers[0], 2 ** 14)
    plt.xticks(2 ** np.linspace(4, 14, 6))
    plt.yticks(np.linspace(0.4, 0.8, num=5))
    plt.xlabel("Super particles $N_\\text{sim}\\,/\\,N_\\text{cell}$ (1)")
    plt.ylabel("Frequency $\\omega_\\text{max}\\,/\\,\\omega_\\text{pp}$ (1)")
    plt.legend([l_theory, (l_vary, l_8192), r_fail],
               ["Theory", "Simulation", "No wave"],
               loc=(0.2535, 0.08), markerscale=0.7, framealpha=0.6,
               handler_map={tuple: matplotlib.legend_handler.HandlerTuple(ndivide=None)})
    if save:
        _saveFigure("frequency-vs-num_particles", "particles_per_cell")

def particleVariationGrowthRate(
    info: RunInfo,
    save: bool=False
):
    time, (energies,), folders = analysis.readFromVariation(
        folder=PARTICLE_VARIATION_FOLDER,
        dataset_names=["/Electric Field/ex"],
        processElement=lambda x: np.mean(np.array(x) ** 2, axis=1),
        time_interval=slice(0,751),
        recursive=True
    )
    with h5py.File(FOLDER_1D / "proton-alpha-instability-1D.h5") as f:
        p8192_time = f["Header/time"][:]
        p8192_energy = np.mean(f["Electric Field/ex"][:] ** 2, axis=1)

    time *= info.omega_pp
    energies *= constants.epsilon_0 / (2.0 * constants.electron_volt)
    p8192_time *= info.omega_pp
    p8192_energy *= constants.epsilon_0 / (2.0 * constants.electron_volt)
    # extract particle numbers
    particle_numbers = np.array([int(pfs[0].stem[-4:]) for pfs in folders])
    # extract growth rates from fits
    fits = [[analysis.fitGrowthRate(time, W_E) for W_E in es] for es in energies]
    p8192_growth_rate = analysis.fitGrowthRate(p8192_time, p8192_energy)[0].slope / 2

    growth_rates = [[res[0].slope / 2 for res in fs if res is not None] for fs in fits]
    growth_rates_mean = np.array([np.mean(x) if len(x) == 4 else np.nan for x in growth_rates])
    growth_rates_std  = np.array([np.std(x)  if len(x) == 4 else np.nan for x in growth_rates])

    plt.style.use(MPLSTYLE_FILE)
    plt.figure(figsize=FIGURE_HALF_SIZE)
    l_vary = plt.errorbar(
        particle_numbers, growth_rates_mean, yerr=growth_rates_std,
        ls="", lw=1.5, ecolor="black", color="white",
        marker="o", markersize=10, markeredgecolor="black", markeredgewidth=1.5,
    )
    l_8192 = plt.plot(
        8192, p8192_growth_rate,
        ls="", lw=1.5, color="white",
        marker="p", markersize=10, markeredgecolor="black", markeredgewidth=1.5
    )[-1]
    l_theory = plt.axhline(theory.growthRate(1e-1), color="black", ls=":")
    plt.ylim(1e-2, 12e-2)
    y_min, y_max = plt.gca().get_ylim()
    r_fail = plt.fill_between(
        [0, 2 ** 6.5], y1=y_min, y2=y_max,
        color="red", alpha=0.7
    )
    plt.xscale("log", base=2)
    plt.xlim(0.5 * particle_numbers[0], 2 ** 14)
    plt.xticks(2 ** np.linspace(4, 14, 6))
    plt.yticks(np.linspace(0.3e-1, 1.2e-1, 4))
    plt.xlabel("Super particles $N_\\text{sim}\\,/\\,N_\\text{cell}$ (1)")
    plt.ylabel("Growth rate $\\gamma\\,/\\,\\omega_\\text{pp}$ (1)")
    plt.legend([l_theory, (l_vary, l_8192), r_fail],
               ["Theory $\\gamma_\\text{max}\\,/\\,\\omega_\\text{pp}$", "Simulation", "No wave"],
               loc=(0.07,0.62), markerscale=0.7,
               handler_map={tuple: matplotlib.legend_handler.HandlerTuple(ndivide=None)})
    if save:
        _saveFigure("growth_rate-vs-num_particles", "particles_per_cell")

def linearTheoryDensityRatio(info: RunInfo, save: bool=False):
    import theory
    with h5py.File("theory_density_ratio.h5") as f:
        na_np = f["na_np_ratio"][:]
        n_p = info.electron.number_density / (2 * na_np + 1)
        omega_pp = physics.plasmaFrequency(
            info.proton.mass,
            info.proton.charge,
            n_p
        )
        k_max = f["k_max"] * info.lambda_D
        omega_max = f["omega_max"] / omega_pp
        gamma_max = f["gamma_max"] / omega_pp
    plt.style.use(MPLSTYLE_FILE)
    plt.figure(figsize=FIGURE_FULL_SIZE)
    plt.plot(na_np, gamma_max, label="$\\gamma_\\text{max}\\,/\\,\\omega_\\text{pp}$", color="cornflowerblue")
    plt.plot(na_np, omega_max, label="$\\omega_\\text{max}\\,/\\,\\omega_\\text{pp}$", color="#888888")
    plt.plot(na_np, k_max, label="$k_\\text{max}\\,\\lambda_\\text{D}$", color="black")
    plt.xscale("log")
    plt.xlim(1e-3, 1e1)
    plt.xlabel("Density ratio $n_\\alpha\\,/\\,n_\\text{p}$ (1)")
    plt.ylabel("Wave properties (1)")
    plt.legend()
    if save:
        generalSaveFigure("linear_theory-density_ratio")

def linearTheoryFlowVelocity(info: RunInfo, save: bool=False):
    with h5py.File("theory_u_alpha_dispersion.h5") as f:
        u_alpha = f["u_alpha_bulk"][:] * 1e-3
        gamma = f["gamma_max"][:] / info.omega_pp
        theta = f["theta_max"][:] * 180 / np.pi
        k_vec = f["k_max"][:] * info.lambda_D
        omega = f["omega_max"][:] / info.omega_pp

    plt.style.use(MPLSTYLE_FILE)
    plt.figure(figsize=FIGURE_FULL_SIZE)
    plt.plot(
        u_alpha, gamma, color="cornflowerblue",
        label="$\\gamma_\\text{max}\\,/\\,\\omega_\\text{pp}$")
    plt.plot(
        u_alpha, omega, color="#888888",
        label="$\\omega_\\text{max}\\,/\\,\\omega_\\text{pp}$")
    plt.plot(
        u_alpha, k_vec, color="black",
        label="$k_\\text{max}\\,\\lambda_\\text{D}$")
    plt.plot(
        u_alpha, theta * np.pi / 180, label="$\\theta_\\text{max}$")
    # plt.xscale("log")
    plt.xlim(55, 200)
    plt.xlabel("Flow velocity $u_\\alpha^{t=0}$ (km$\\,/\\,$s)")
    plt.ylabel("Wave properties (1)")
    plt.legend(loc=(0.7, 0.25))
    if save:
        generalSaveFigure("linear_theory-flow_velocity")

def linearTheoryWaveProperties(info: RunInfo, save: bool=False):
    plt.style.use(MPLSTYLE_FILE)
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_FULL_SIZE[0], 3.2), sharey=True)
    axes: list[plt.Axes] = axes

    with h5py.File("theory_density_ratio.h5") as f:
        na_np = f["na_np_ratio"][:]
        n_p = info.electron.number_density / (2 * na_np + 1)
        omega_pp = physics.plasmaFrequency(
            info.proton.mass,
            info.proton.charge,
            n_p
        )
        k_max = f["k_max"] * info.lambda_D
        omega_max = f["omega_max"] / omega_pp
        gamma_max = f["gamma_max"] / omega_pp

    axes[0].plot(na_np, gamma_max, label="$\\gamma_\\text{max}\\,/\\,\\omega_\\text{pp}$", color="cornflowerblue")
    axes[0].plot(na_np, omega_max, label="$\\omega_\\text{max}\\,/\\,\\omega_\\text{pp}$", color="#888888")
    axes[0].plot(na_np, k_max, label="$k_\\text{max}\\,\\lambda_\\text{D}$", color="black")
    axes[0].set(
        xscale="log",
        xlim=(1e-3, 1e1),
        xlabel="Density ratio $n_\\alpha\\,/\\,n_\\text{p}$ (1)",
        ylabel="Wave properties (1)",
    )
    axes[0].legend(labelspacing=0.4, loc=(0.05, 0.15))

    with h5py.File("theory_u_alpha_dispersion.h5") as f:
        u_alpha = f["u_alpha_bulk"][:] * 1e-3
        gamma = f["gamma_max"][:] / info.omega_pp
        theta = f["theta_max"][:] * 180 / np.pi
        k_vec = f["k_max"][:] * info.lambda_D
        omega = f["omega_max"][:] / info.omega_pp

    axes[1].plot(
        u_alpha, gamma, color="cornflowerblue",
        label="$\\gamma_\\text{max}\\,/\\,\\omega_\\text{pp}$")
    axes[1].plot(
        u_alpha, omega, color="#888888",
        label="$\\omega_\\text{max}\\,/\\,\\omega_\\text{pp}$")
    axes[1].plot(
        u_alpha, k_vec, color="black",
        label="$k_\\text{max}\\,\\lambda_\\text{D}$")
    axes[1].plot(
        u_alpha, theta * np.pi / 180,
        label="$\\theta_\\text{max}$",
        color="orange", ls="-."
    )
    # magic = np.arccos((omega / k_vec * info.omega_pp * info.lambda_D + 30_000) / (u_alpha * 1e3))
    # print(magic)
    # axes[1].plot(u_alpha, magic, color="black")
    axes[1].set(
        xlim=(55, 200),
        xlabel="Flow velocity $u_\\alpha^{t=0}$ (km$\\,/\\,$s)",
        # ylabel="Wave properties (1)",
    )
    axes[1].legend(loc=(0.48, 0.18), labelspacing=0.4)
    axes[0].text(
        0.05, 0.95, s="$\\mathbf{(a)}$",
        horizontalalignment="left",
        verticalalignment="top",
        transform=axes[0].transAxes,
    )
    axes[1].text(
        0.05, 0.95, s="$\\mathbf{(b)}$",
        horizontalalignment="left",
        verticalalignment="top",
        transform=axes[1].transAxes,
    )
    plt.tight_layout(w_pad=-0.5)
    if save:
        generalSaveFigure("linear_theory-wave_properties")

def illustrateVelocitySpace(info: RunInfo, save: bool=False):
    v_ph = 69
    v_th = np.sqrt(3/2) * info.alpha.v_thermal * 1e-3
    u_alpha = 120
    theta = np.arccos((v_ph + v_th) / u_alpha, out=np.array(0.0), where=v_ph / (u_alpha - v_th) < 1)
    # phase velocity circle and gradient circle
    alpha = np.linspace(0, 2 * np.pi, num=100)
    plt.style.use(MPLSTYLE_FILE)
    plt.figure(figsize=FIGURE_FULL_SIZE)
    plt.plot(v_ph * np.sin(alpha), v_ph * np.cos(alpha), ls=":", lw=2, label="$\\mathbf{v}\\cdot\\mathbf{v}_\\text{ph}=v_\\text{ph}^2$", color="#000000")
    plt.plot(u_alpha + v_th * np.sin(alpha), v_th * np.cos(alpha), ls=(0, (3,1,1,1)), lw=2, label="$\\mathbf{v}\\cdot\\mathbf{u}_\\alpha=v_{\\text{th},\\alpha}^2$", color="#2C2CEA")
    alpha = np.linspace(-theta, theta)
    plt.plot(0.5 * v_ph * np.cos(alpha), 0.5 * v_ph * np.sin(alpha), color="black", ls="-", lw=1.5)
    plt.text(
        v_ph / 4, 0, s=r"$2\theta$",
        horizontalalignment="left",
        verticalalignment="center",
    )
    # proton and alpha max
    plt.plot(0,0, ls="", marker="o", color="black", label="$\\langle f_\\text{p}\\rangle$")
    plt.plot(u_alpha, 0, ls="", marker="p", color="#2C2CEA", label="$\\langle f_\\alpha\\rangle$")
    # Interaction center
    s = np.array([-1_000, 1_000])
    plt.plot(
        v_ph * np.cos(theta) + np.sin(theta) * s,
        v_ph * np.sin(theta) - np.cos(theta) * s,
        ls="--", color="#900000"
    )
    plt.plot(
        +v_ph * np.cos(theta) + np.sin(theta) * s,
        -v_ph * np.sin(theta) + np.cos(theta) * s,
        ls="--", color="#900000"
    )
    # interaction regions
    width = 40
    rect_pos = plt.Rectangle(
        xy=(
            (v_ph - width / 2) * np.cos(theta) + np.sin(theta) * (-2 * v_ph),
            (v_ph - width / 2) * np.sin(theta) - np.cos(theta) * (-2 * v_ph)
        ),
        width=1000, height=width, angle=theta * 180 / np.pi+270, edgecolor="black", zorder=1, facecolor="#dc7800", alpha=0.7
    )
    rect_neg = plt.Rectangle(
        xy=(
            +(v_ph + width / 2) * np.cos(theta) + np.sin(theta) * (-2 * v_ph),
            -(v_ph + width / 2) * np.sin(theta) + np.cos(theta) * (-2 * v_ph)
        ),
        width=1000, height=width, angle=-theta * 180 / np.pi+90, edgecolor="black", zorder=1, facecolor="#dc7800", alpha=0.7
    )
    ann = plt.annotate(
        text='', xy=((v_ph + 3) * np.cos(theta), (v_ph + 3) * np.sin(theta)),
        xytext=(0,0), arrowprops=dict(arrowstyle='->', lw=2)
    )
    ann = plt.annotate(
        text='', xy=((v_ph + 3) * np.cos(theta), -(v_ph + 3) * np.sin(theta)),
        xytext=(0,0), arrowprops=dict(arrowstyle='->', lw=2)
    )

    delta = 120
    ann = plt.annotate(
        text='', xy=(
            (v_ph - width/2 - 3) * np.cos(theta) + np.sin(theta) * delta,
            (v_ph - width/2 - 3) * np.sin(theta) - np.cos(theta) * delta
        ),
        xytext=(
            (v_ph + width/2 + 3) * np.cos(theta) + np.sin(theta) * delta,
            (v_ph + width/2 + 3) * np.sin(theta) - np.cos(theta) * delta
        ), arrowprops=dict(arrowstyle='<->', lw=2, color="#693a00")
    )
    ann = plt.annotate(
        text='', xy=(
            (v_ph - width/2 - 3) * np.cos(theta) + np.sin(theta) * delta,
            -(v_ph - width/2 - 3) * np.sin(theta) + np.cos(theta) * delta
        ),
        xytext=(
            (v_ph + width/2 + 3) * np.cos(theta) + np.sin(theta) * delta,
            -(v_ph + width/2 + 3) * np.sin(theta) + np.cos(theta) * delta
        ), arrowprops=dict(arrowstyle='<->', lw=2, color="#693a00")
    )
    plt.text(
        0.02, 0.97, s=r"$u_\alpha = v_\text{ph} + v_{\text{th},\alpha}$",
        horizontalalignment="left",
        verticalalignment="top",
        transform=plt.gca().transAxes,
    )
    plt.gca().add_patch(rect_pos)
    plt.gca().add_patch(rect_neg)
    plt.gca().set_aspect("equal")
    plt.xlim(-80, 160)
    plt.ylim(-80, 80)
    plt.xlabel(r"Velocity $v_x/v_\text{ph}$")
    plt.ylabel(r"Velocity $v_y/v_\text{ph}$")
    plt.xticks([-v_ph, 0, v_ph, 2*v_ph], ["-1", "0", "1", "2"])
    plt.yticks([-v_ph, 0, v_ph], ["-1", "0", "1"])
    [h, l] = plt.gca().get_legend_handles_labels()
    plt.legend(
        [(h + [
            plt.scatter(1e3, 1e3, c='black', marker='$\\longrightarrow$', s=300),
            plt.scatter(1e3, 1e3, c='#693a00', marker='$⟷$', s=300)
        ])[i] for i in (0,1,4,2,3,5)],
        [(l+["$\\mathbf{v}_\\text{ph}$", r"$\partial f_s\,/\,\partial t$"])[i] for i in (0,1,4,2,3,5)],
        ncols=2, labelspacing=0.0, loc="lower left", columnspacing=0.5)
    if save:
        generalSaveFigure("velocity_space-illustration")

def illustrateSimulationGrid(save: bool=False):
    rng = np.random.default_rng(2)
    plt.style.use(MPLSTYLE_FILE)
    plt.figure(figsize=(FIGURE_FULL_SIZE[0], 3))
    plt.plot(0.1 + rng.random(100) * 3.8, rng.random(100) * 2, ls="", marker="o", markersize=3, color="#aaaaaa")
    plt.plot(3.15, 0.85, ls="", marker="o", markersize=5, markeredgecolor="black", markeredgewidth=1, color="#666666")
    plt.arrow(3.15, 0.85, -0.27, 0.25, head_width=0.04, width=0.01, color="black", length_includes_head=True)
    plt.text(
        2.85, 1.17, "$\\mathbf{v}_\\text{n}\\,\\Delta t$",
        horizontalalignment="left", verticalalignment="center")
    plt.text(
        3.24, 0.93, "$\\mathbf{x}_\\text{n}$",
        horizontalalignment="center", verticalalignment="center")
    # plt.fill_between([2/3, 4/3],[4/3, 4/3], [6/3, 6/3], zorder=2, alpha=0.6, color="darkorange")
    for i in np.linspace(0, 4, 7):
        plt.plot([i,i], [0,2], ls="-", color="black")
    for j in np.linspace(0, 2, 4):
        plt.plot([0,4], [j,j], ls="-", color="black")
    plt.arrow(
        -0.08,0,0,2/3, head_width=0.04, width=0.01,
        color="black", length_includes_head=True)
    plt.text(
        -0.08,2/3/2, s="$\\Delta y$", rotation=90,
        horizontalalignment="right", verticalalignment="center")
    plt.arrow(0,-0.08,2/3,0, head_width=0.04, width=0.01,
        color="black", length_includes_head=True)
    plt.text(
        2/3/2,-0.12, s="$\\Delta x$",
        horizontalalignment="center", verticalalignment="top")
    plt.plot(*np.array([
            (i,j) for i in np.linspace(0, 4, 7) for j in np.linspace(0, 2, 4)
        ]).T,
        color="#3333FF", ls="", marker="s", markersize=4, zorder=5)
    plt.text(0, 2.05,
        s="$\\mathbf{E}_\\text{n},\\mathbf{B}_\\text{n},\\mathbf{J}_\\text{n}$",
        horizontalalignment="center", verticalalignment="bottom", color="#3333ff")
    plt.text(-0.2, 4/3, "$\\mathbf{\\vdots}$", horizontalalignment="center", verticalalignment="center", color="#3333ff")
    plt.text(2/3, 2.05, "$\\mathbf{\\dots}$", horizontalalignment="center", verticalalignment="bottom", color="#3333ff")
    plt.text(2/3-0.1, 4/3+0.07, "$\\mathbf{\\ddots}$", horizontalalignment="right", verticalalignment="bottom", color="#3333ff")
    plt.gca().set_axis_off()
    plt.gca().set_aspect("equal")
    if save:
        generalSaveFigure("simulation_grid-illustration")
