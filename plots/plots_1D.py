from pathlib import Path

import colormaps as cmaps
import matplotlib.legend_handler
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy import constants, optimize, signal, io
import h5py

import analysis
import theory
from basic import physics, Distribution
from .general import generalSaveFigure, plotEnergyEFieldOverTime, _loadSpaceMomDistribution

from basic.paths import (
    RESULTS_FOLDER,
    FOLDER_1D,
    PARTICLE_VARIATION_FOLDER,
    DENSITY_VARIATION_FOLDER,
    THEORY_DENSITY_RATIO_FILE,
    THEORY_U_ALPHA_FILE,
    MPLSTYLE_FILE,
)
from plots.settings import FIGURE_HALF_SIZE, FIGURE_FULL_SIZE
from basic import RunInfo, Species, SpeciesInfo

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
    cax = plt.colorbar(label="Electric field $E_x$ (V/m)")
    cax.ax.set_yticks(np.linspace(-1, 1, 5,))
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
    print(temperature[0], temperature[-1])
    plt.style.use(MPLSTYLE_FILE)
    if species == Species.ELECTRON:
        plt.figure(figsize=(FIGURE_HALF_SIZE[0]-0.26, 2.5))
    else:
        plt.figure(figsize=(FIGURE_HALF_SIZE[0], 2.5))
    plt.plot(time, temperature, label=f"$T_{{{species.symbol()},x}}$")
    plt.plot(time, temp_3d, label=f"$T_{{{species.symbol()}}}$", color="#aaaaaa")
    plt.xlabel("Time $t\\,\\omega_\\text{pp}$ (1)")
    plt.ylabel(f"Temperature $T$ (eV)")
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
            xlabel = f"$v_{species.symbol()}\\,/\\,v^{{t=0}}_{{\\text{{t}}{species.symbol()}}}$ (1)",
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
        label=r"Distribution $\langle f_{s}\rangle_x$ (a.u.)")
    axes[1,0].set(
        ylim = (0,150),
        yticks = np.linspace(0, 140, 8),
        ylabel=r"Time $t\,\omega_\text{pp}$ (1)",
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
        ylabel=r"$\langle f_{s}\rangle_x$ (a.u.)",
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

def eFieldEvolutionCombined(filename: Path, info: RunInfo, save: bool=False):
    filename = PARTICLE_VARIATION_FOLDER / "particles_8192/rep_0.h5"
    with h5py.File(filename) as f:
        time = f["Header/time"][:] * info.omega_pp
        grid_edges = f["Grid/grid"][:]
        ex = f["Electric Field/ex"][:]

    grid_edges = grid_edges[0] / info.lambda_D

    plt.style.use(MPLSTYLE_FILE)
    fig, axes = plt.subplots(2, 1, figsize=(FIGURE_FULL_SIZE[0], 5.5), sharex=True)
    axes: list[plt.Axes] = axes
    quad = axes[0].pcolormesh(time, grid_edges, ex[:-1].T, cmap="bwr", rasterized=True,
                vmin=-np.max(np.abs(ex)), vmax=np.max(np.abs(ex)))
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(quad, cax=cax, orientation='vertical', label="Electric field $E_x$ (V/m)")
    axes[0].set(
        # xlabel="Time $t\\,\\omega_\\text{pp}$ (1)"
        ylabel="Position $x\\,/\\,\\lambda_\\text{D}$ (1)",
        yticks=np.arange(5) * 32,
        ylim=(0, np.max(grid_edges)),
        xlim=(0, 150),
        xticks=np.linspace(0, 150, 6),
    )

    with h5py.File(filename) as f:
            time = f["Header/time"][:] * info.omega_pp
            energy = f["Electric Field/ex"][:] ** 2
    energy = np.mean(
        energy,
        axis=tuple(range(1, energy.ndim))
    ) * (constants.epsilon_0 / 2) / (constants.electron_volt)
    fit_result = analysis.fitGrowthRate(time, energy)

    axes[1].plot(time, energy, label="$\\langle W_E\\rangle_x^\\text{sim}$",
            color="black", lw=2)

    lin_fit, fit_interval, poly_info = fit_result
    axes[1].plot(
            time[slice(*fit_interval)], energy[slice(*fit_interval)],
            color="orange", lw=3, ls="solid", zorder=3,
            label="Linear regime",
    )
    axes[1].plot(
            time, np.exp(lin_fit.slope * time + lin_fit.intercept),
            ls="--", color="royalblue", zorder=9, lw=1.5,
            label="$W_{E}\\propto\\exp(2\\gamma\\,t)$",
    )
    poly, extrema, turn_p = poly_info
    axes[1].plot(
        time, np.exp(poly(time)), label="Polynomial fit",
        color="gray", lw=1.5, ls=":", zorder=2, alpha=0.8
    )
    axes[1].plot(
        extrema, np.exp(poly(extrema)), label="Turning points",
        color="gray", zorder=3, ls="", alpha=0.8,
        marker="o", markeredgecolor="black", markeredgewidth=1.5, markersize=10,
    )
    axes[1].plot(
        turn_p, np.exp(poly(turn_p)), label="Inflection point",
        color="white", zorder=3, ls="", marker="p",
        markersize=10, markeredgecolor="black", markeredgewidth=1.5,
    )
    axes[1].set(
        xlim=(0, 150),
        ylim=(1e4, 1e7),
        xticks=np.linspace(0.0, 150.0, num=6),
        yscale="log",
        xlabel="Time $t\\,\\omega_\\text{pp}$ (1)",
        ylabel="Energy $\\langle W_E\\rangle_x$ (eV$\\,/\\,$m$^3$)",
    )
    axes[1].legend(columnspacing=1, labelspacing=0.4)
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cax.set_axis_off()
    axes[0].text(
        0.05, 0.95, s="$\\mathbf{(a)}$",
        horizontalalignment="left",
        verticalalignment="top",
        transform=axes[0].transAxes
    )
    axes[1].text(
        0.05, 0.95, s="$\\mathbf{(b)}$",
        horizontalalignment="left",
        verticalalignment="top",
        transform=axes[1].transAxes
    )
    fig.tight_layout(h_pad=0)
    if save:
        _saveFigure("e_field_evolution")


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
                marker="o", markeredgecolor="black", markeredgewidth=1.5)
    l_8192 = plt.plot(8192, p8192_T_diff,
        ls="", lw=1.5, color="white",
        marker="p", markeredgecolor="black", markeredgewidth=1.5)[-1]
    plt.xscale("log", base=2)
    plt.xlabel("Pseudoparticles $N_\\text{sim}\\,/\\,N_\\text{cell}$ (1)")
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

    with h5py.File(PARTICLE_VARIATION_FOLDER / "particles_8192/rep_0.h5") as f:
        p8192_time = f["Header/time"][:]
        p8192_E_field = f["Electric Field/ex"][:]
        p8192_grid = np.squeeze(f["Grid/grid"])

    # fix units of time and energy
    time *= info.omega_pp
    grid /= info.lambda_D
    p8192_time *= info.omega_pp
    p8192_grid /= info.lambda_D
    # extract particle numbers
    particle_numbers = np.array([int(pfs[0].stem[-4:]) for pfs in folders])
    k_arr = np.full(shape=E_fields.shape[:2], fill_value=np.nan)
    k_err_arr = np.full_like(k_arr, fill_value=np.nan)

    for p_idx, e_fields in enumerate(E_fields):
        if p_idx < 2:
            continue
        for r_idx, field in enumerate(e_fields):
            _, regime, _ = analysis.fitGrowthRate(
                time, np.mean(field ** 2, axis=1), allowed_slope_deviation=0.5)
            k_arr[p_idx, r_idx], k_err_arr[p_idx, r_idx] = analysis.estimateFrequency(
                -1, grid, field[:(regime[-1])]
            )
    _, regime, _ = analysis.fitGrowthRate(
        p8192_time, np.mean(p8192_E_field ** 2, axis=1)
    )
    p8192_k, p8192_k_err = analysis.estimateFrequency(
        -1, p8192_grid, p8192_E_field[:regime[-1]]
    )

    mean_k = np.mean(k_arr, axis=-1)
    mean_k_err = np.sqrt(np.sum((k_err_arr / 4) ** 2, axis=-1) + np.var(k_arr, axis=-1) / 4)
    p8192_k_err = np.sqrt(p8192_k_err ** 2 + np.mean(np.var(k_arr, axis=1)[-2:]))

    plt.style.use(MPLSTYLE_FILE)
    plt.figure(figsize=FIGURE_HALF_SIZE)
    l_vary = plt.errorbar(particle_numbers[2:], mean_k[2:], yerr=mean_k_err[2:],
        ls="", lw=1.5, marker="o", color="white", ecolor="black",
        markeredgecolor="black", markeredgewidth=1.5)
    l_8192 = plt.errorbar(8192, p8192_k, yerr=p8192_k_err,
        ls="", lw=1.5, color="white", ecolor="black",
        marker="p", markeredgecolor="black", markeredgewidth=1.5)

    l_theory = plt.axhline(0.6604529941471382, color="black", ls=":")
    plt.ylim(0.15, 0.75)
    y_min, y_max = plt.gca().get_ylim()
    r_fail = plt.fill_between(
        [0, 2 ** 6.5], y1=y_min, y2=y_max,
        color="red", alpha=0.7
    )
    plt.xscale("log", base=2)
    plt.xlim(0.5 * particle_numbers[0], 2 ** 14)
    plt.xticks(2 ** np.linspace(4, 14, 6))
    plt.ylim(y_min, y_max)
    plt.yticks(np.linspace(0.2, 0.7, num=6))
    plt.xlabel("Pseudoparticles $N_\\text{sim}\\,/\\,N_\\text{cell}$ (1)")
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
    with h5py.File(PARTICLE_VARIATION_FOLDER / "particles_8192/rep_0.h5") as f:
        p8192_time = f["Header/time"][:]
        p8192_E_field = f["Electric Field/ex"][:]

    # fix units of time and energy
    time *= info.omega_pp
    p8192_time *= info.omega_pp
    # extract particle numbers
    particle_numbers = np.array([int(pfs[0].stem[-4:]) for pfs in folders])
    omega_arr = np.full(shape=E_fields.shape[:2], fill_value=np.nan)
    omega_err_arr = np.full_like(omega_arr, fill_value=np.nan)

    for p_idx, e_fields in enumerate(E_fields):
        if p_idx < 2:
            continue
        for r_idx, field in enumerate(e_fields):
            _, regime, _ = analysis.fitGrowthRate(
                time, np.mean(field ** 2, axis=1), allowed_slope_deviation=0.5
            )
            omega_arr[p_idx, r_idx], omega_err_arr[p_idx, r_idx] = analysis.estimateFrequency(
                -2, time, field[:(regime[-1])]
            )
    _, regime, _ = analysis.fitGrowthRate(
        p8192_time, np.mean(p8192_E_field ** 2, axis=1)
    )
    p8192_omega, p8192_omega_err = analysis.estimateFrequency(
        -2, p8192_time, p8192_E_field[:regime[-1]]
    )
    mean_omega = np.mean(omega_arr, axis=1)
    mean_omega_err = np.sqrt(np.sum((omega_err_arr / 4) ** 2, axis=1) + np.var(omega_arr, axis=1) / 4)
    p8192_omega_err = np.sqrt(p8192_omega_err ** 2 + np.mean(np.var(omega_arr, axis=1)[-2:]))
    colors = cmaps.devon.discrete(3).colors
    plt.style.use(MPLSTYLE_FILE)
    plt.figure(figsize=FIGURE_HALF_SIZE)
    l_vary = plt.errorbar(particle_numbers[2:], mean_omega[2:], yerr=mean_omega_err[2:],
        ls="", marker="o", color=colors[2], lw=1.5,
        markeredgecolor="black", ecolor="black", markeredgewidth=1.5)
    l_8192 = plt.errorbar(2 ** 13, p8192_omega, yerr=p8192_omega_err,
        ls="", marker="p", color=colors[2], lw=1.5,
        markeredgecolor="black", ecolor="black", markeredgewidth=1.5)

    with h5py.File(THEORY_U_ALPHA_FILE) as f:
        u_alpha = f["u_alpha_bulk"][:] * 1e-3
        omega = f["omega_max"][:] / info.omega_pp
    l_theory = plt.axhline(omega[np.argmin(np.abs(u_alpha - 100))], color="black", ls=":")
    plt.ylim(0.7, 0.95)
    y_min, y_max = plt.gca().get_ylim()
    r_fail = plt.fill_between(
        [0, 2 ** 6.5],
        y1=y_min, y2=y_max,
        color="red", alpha=0.7
    )
    plt.xscale("log", base=2)
    plt.xlim(0.5 * particle_numbers[0], 2 ** 14)
    plt.xticks(2 ** np.linspace(4, 14, 6))
    plt.yticks(np.linspace(0.7, 0.9, num=3))
    plt.xlabel("Pseudoparticles $N_\\text{sim}\\,/\\,N_\\text{cell}$ (1)")
    plt.ylabel("Frequency $\\omega_\\text{max}\\,/\\,\\omega_\\text{pp}$ (1)")
    plt.legend([l_theory, (l_vary, l_8192), r_fail],
               ["Theory", "Simulation", "No wave"],
               loc="upper left", markerscale=0.7, framealpha=0.6,
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
    p8192_growth_rate_err = np.abs(
        analysis.fitGrowthRate(p8192_time, p8192_energy, allowed_slope_deviation=0.4)[0].slope -
        analysis.fitGrowthRate(p8192_time, p8192_energy, allowed_slope_deviation=0.1)[0].slope
    ) / 2
    growth_rates = np.full(energies.shape[:2], np.nan)
    growth_rate_errs = np.full_like(growth_rates, np.nan)
    for p_idx, es in enumerate(energies):
        for r_idx, W_E in enumerate(es):
            res = analysis.fitGrowthRate(time, W_E)
            res_small = analysis.fitGrowthRate(time, W_E, allowed_slope_deviation=0.1)
            res_big = analysis.fitGrowthRate(time, W_E, allowed_slope_deviation=0.4)
            if None not in [res, res_small, res_big]:
                growth_rates[p_idx,r_idx] = res[0].slope / 2
                growth_rate_errs[p_idx,r_idx] = np.abs(res_small[0].slope - res_big[0].slope) / 2

    growth_rates_mean = np.mean(growth_rates, axis=1)
    growth_rates_err  = np.sqrt(np.sum((growth_rate_errs / 4) ** 2, axis=1) + np.var(growth_rates, axis=1) / 4)
    p8192_growth_rate_err = np.sqrt(p8192_growth_rate_err ** 2 + np.mean(np.var(growth_rates, axis=1)[-2:]))
    plt.style.use(MPLSTYLE_FILE)
    plt.figure(figsize=FIGURE_HALF_SIZE)
    l_vary = plt.errorbar(
        particle_numbers, growth_rates_mean, yerr=growth_rates_err,
        ls="", lw=1.5, ecolor="black", color="white",
        marker="o", markersize=10, markeredgecolor="black", markeredgewidth=1.5,
    )
    l_8192 = plt.errorbar(
        8192, p8192_growth_rate, yerr=p8192_growth_rate_err,
        ls="", lw=1.5, color="white",
        marker="p", markersize=10, markeredgecolor="black", markeredgewidth=1.5
    )
    with h5py.File(THEORY_DENSITY_RATIO_FILE) as f:
        na_np = f["na_np_ratio"][:]
        idx = np.argmin(np.abs(na_np- 1e-1))
        theory_info = runInfoForDenistyRatio(na_np[idx])
        gamma_max = f["gamma_max"][idx] / theory_info.omega_pp
    l_theory = plt.axhline(gamma_max, color="black", ls=":")
    plt.ylim(1e-2, 13e-2)
    y_min, y_max = plt.gca().get_ylim()
    r_fail = plt.fill_between(
        [0, 2 ** 6.5], y1=y_min, y2=y_max,
        color="red", alpha=0.7
    )
    plt.xscale("log", base=2)
    plt.xlim(0.5 * particle_numbers[0], 2 ** 14)
    plt.xticks(2 ** np.linspace(4, 14, 6))
    plt.yticks(np.linspace(0.3e-1, 1.2e-1, 4))
    plt.xlabel("Pseudoparticles $N_\\text{sim}\\,/\\,N_\\text{cell}$ (1)")
    plt.ylabel("Growth rate $\\gamma\\,/\\,\\omega_\\text{pp}$ (1)")
    plt.legend([l_theory, (l_vary, l_8192), r_fail],
               ["Theory $\\gamma_\\text{max}\\,/\\,\\omega_\\text{pp}$", "Simulation", "No wave"],
               loc=(0.07,0.62), markerscale=0.7,
               handler_map={tuple: matplotlib.legend_handler.HandlerTuple(ndivide=None)})
    if save:
        _saveFigure("growth_rate-vs-num_particles", "particles_per_cell")

def linearTheoryDensityRatio(info: RunInfo, save: bool=False):
    import theory
    with h5py.File(THEORY_DENSITY_RATIO_FILE) as f:
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
    with h5py.File(THEORY_U_ALPHA_FILE) as f:
        u_alpha = f["u_alpha_bulk"][:] * 1e-3
        gamma = f["gamma_max"][:] / info.omega_pp
        theta = f["theta_max"][:] * 180 / np.pi
        k_vec = f["k_max"][:] * info.lambda_D
        omega = f["omega_max"][:] / info.omega_pp
    print(u_alpha[0], np.mean((omega/k_vec * info.omega_pp*info.lambda_D)[-10:]), np.sqrt(3/2)*info.alpha.v_thermal)
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

    with h5py.File(THEORY_DENSITY_RATIO_FILE) as f:
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

    with h5py.File(THEORY_U_ALPHA_FILE) as f:
        u_alpha = f["u_alpha_bulk"][:] * 1e-3
        gamma = f["gamma_max"][:] / info.omega_pp
        theta = f["theta_max"][:]
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
        u_alpha, theta,
        label="$\\theta_\\text{max}$",
        color="orange", ls="-.", lw=2,
    )
    v_ph_magic = (omega / k_vec)[u_alpha>100] * info.omega_pp * info.lambda_D
    popt, _ = optimize.curve_fit(
        lambda v, d: (np.mean(v_ph_magic) + d) / v,
        xdata=u_alpha[u_alpha>100] * 1e3,
        ydata=np.cos(theta[u_alpha>100]), p0=[np.sqrt(np.pi/2) * info.alpha.v_thermal]
    )
    magic = np.arccos(
        (np.mean(v_ph_magic) + popt[0]) / (u_alpha * 1e3),
        out=np.zeros_like(omega),
        where=(np.mean(v_ph_magic) + popt[0]) / (u_alpha * 1e3) < 1
    )
    axes[1].plot(
        u_alpha, magic, color="#bb0000", ls=(0,(1,1,1,5)),
        label=r"$\theta_\text{max}^\text{geom}$",
        lw=1.5
    )
    axes[1].set(
        xlim=(55, 200),
        xlabel="Flow velocity $u_\\alpha^{t=0}$ (km$\\,/\\,$s)",
        # ylabel="Wave properties (1)",
    )
    axes[1].legend(loc=(0.48, 0.03), labelspacing=0.4, framealpha=0.6)
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

def illustrateVelocitySpace(info: RunInfo, u_alpha: float|None=None, save: bool=False):
    v_ph = 69
    v_th = np.sqrt(3/2) * info.alpha.v_thermal * 1e-3
    u_crit = v_ph + v_th
    if u_alpha is None:
        u_alpha = u_crit
    assert u_alpha >= u_crit, "negative not supported"
    theta = np.arccos(u_crit / u_alpha, out=np.array(0.0), where=v_ph / (u_alpha - v_th) < 1)
    # phase velocity circle and gradient circle
    alpha = np.linspace(0, 2 * np.pi, num=100)
    plt.style.use(MPLSTYLE_FILE)
    plt.figure(figsize=FIGURE_FULL_SIZE)
    plt.plot(
        v_ph * np.sin(alpha), v_ph * np.cos(alpha), ls=":", lw=2,
        label="$\\mathbf{v}\\cdot\\mathbf{v}_\\text{ph}=v_\\text{ph}^2$", color="#000000"
    )
    plt.plot(
        u_alpha + v_th * np.sin(alpha), v_th * np.cos(alpha), ls=(0, (3,1,1,1)), lw=2,
        label="$\\mathbf{v}\\cdot\\mathbf{u}_\\alpha=v_{\\text{r}\\alpha}^2$", color="#2C2CEA"
    )
    alpha = np.linspace(-theta, theta)
    plt.plot(0.5 * v_ph * np.cos(alpha), 0.5 * v_ph * np.sin(alpha), color="black", ls="-", lw=1.5)
    if u_alpha > u_crit:
        plt.text(
            v_ph / 6, 0,
            s=r"$2\theta_\text{max}$",
            horizontalalignment="left",
            verticalalignment="center",
        )
    else:
        plt.text(
            v_ph / 6, v_ph / 8,
            s=r"$\theta_\text{max}=0$",
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
    if u_alpha > u_crit:
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
    if u_alpha > u_crit:
        rect_neg = plt.Rectangle(
            xy=(
                +(v_ph + width / 2) * np.cos(theta) + np.sin(theta) * (-2 * v_ph),
                -(v_ph + width / 2) * np.sin(theta) + np.cos(theta) * (-2 * v_ph)
            ),
            width=1000, height=width, angle=-theta * 180 / np.pi+90, edgecolor="black", zorder=1, facecolor="#dc7800", alpha=0.7
        )
    # arrows (v_ph)
    ann = plt.annotate(
        text='', xy=((v_ph + 3) * np.cos(theta), (v_ph + 3) * np.sin(theta)),
        xytext=(0,0), arrowprops=dict(arrowstyle='->', lw=2)
    )
    if u_alpha > u_crit:
        ann = plt.annotate(
            text='', xy=((v_ph + 3) * np.cos(theta), -(v_ph + 3) * np.sin(theta)),
            xytext=(0,0), arrowprops=dict(arrowstyle='->', lw=2)
        )
    # arrows (interaction)
    if u_alpha > u_crit:
        delta = 120
    else:
        delta = 60
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
        0.02, 0.97,
        s=rf"$u_\alpha {'=' if u_alpha == u_crit else '>'} v_\text{{ph}} + v_{{\text{{r}}\alpha}}$",
        horizontalalignment="left",
        verticalalignment="top",
        transform=plt.gca().transAxes,
    )
    plt.gca().add_patch(rect_pos)
    if u_alpha > u_crit:
        plt.gca().add_patch(rect_neg)
    plt.gca().set_aspect("equal")
    plt.xlim(-80, 160)
    plt.ylim(-80, 80)
    plt.xlabel(r"Velocity $v_x\,/\,v_\text{ph}$")
    plt.ylabel(r"Velocity $v_y\,/\,v_\text{ph}$")
    plt.xticks([-v_ph, 0, v_ph, 2*v_ph], ["-1", "0", "1", "2"])
    plt.yticks([-v_ph, 0, v_ph], ["-1", "0", "1"])
    [h, l] = plt.gca().get_legend_handles_labels()
    plt.legend(
        [(h + [
            plt.scatter(1e3, 1e3, c='black', marker='$\\longrightarrow$', s=300),
            plt.scatter(1e3, 1e3, c='#693a00', marker='$⟷$', s=300)
        ])[i] for i in (0,1,4,2,3,5)],
        [(l+["$\\mathbf{v}_\\text{ph}$", r"$\partial f_s\,/\,\partial t$"])[i] for i in (0,1,4,2,3,5)],
        ncols=2, labelspacing=0.2, loc="lower left", columnspacing=0.5, borderaxespad=0.8)
    if save:
        generalSaveFigure(f"velocity_space{'_crit' if u_alpha == u_crit else ''}-illustration")

def illustrateSimulationGrid(save: bool=False):
    rng = np.random.default_rng(2)
    plt.style.use(MPLSTYLE_FILE)
    plt.figure(figsize=(FIGURE_FULL_SIZE[0], 3))
    plt.plot(0.1 + rng.random(100) * 3.8, rng.random(100) * 2, ls="", marker="o", markersize=3, color="#aaaaaa", markeredgewidth=0)
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

def runInfoForDenistyRatio(ratio):
    n_electron = 12e6
    n_proton = n_electron / (1 + 2 * ratio)
    n_alpha = ratio * n_proton
    return RunInfo(
        electron=SpeciesInfo(
            number_density=12.0e6,
            temperature=100.0,
            charge=-1,
            mass=1.0,
            bulk_velocity=0.0
        ),
        proton=SpeciesInfo(
            number_density=n_proton,
            temperature=3.0,
            charge=+1,
            mass=1836.152674,
            bulk_velocity=0.0
        ),
        alpha=SpeciesInfo(
            number_density=n_alpha,
            temperature=12.0,
            charge=+2,
            mass=7294.29953,
            bulk_velocity=1.0e5
        )
    )

def heatingVsDensityRatio(species: Species, save: bool=False):
    density_ratios = []
    temperature = []
    std_temperature = []

    for filename in sorted(DENSITY_VARIATION_FOLDER.glob("*.h5")):
        ratio = 10 ** float(filename.stem[-5:])
        info = runInfoForDenistyRatio(ratio)
        with h5py.File(filename) as f:
            time = f[f"Header/time"][1:] * info.omega_pp
            x_grid = f[f"Grid/x_px/{species.value}/X"][1:]
            px_grid = f[f"Grid/x_px/{species.value}/Px"][1:]
            temp_x = np.mean(f[f"dist_fn/x_px/{species.value}"][1:], axis=1)
            energy = f["Electric Field/ex"][1:] ** 2
        energy = np.mean(
            energy,
            axis=tuple(range(1, energy.ndim))
        ) * (constants.epsilon_0 / 2) / (constants.electron_volt)
        fit_result = analysis.fitGrowthRate(time, energy, allowed_slope_deviation=0.1)
        # take the same amount of time after the linear regime for each record
        last_idx = (fit_result[1][1] + 880)
        info = runInfoForDenistyRatio(ratio)
        temp = analysis.temperature1D(x_grid[:last_idx], px_grid[:last_idx], temp_x[:last_idx], info[species])
        T_init = np.mean(temp[:10], axis=-1)
        T_final = np.mean(temp[-10:], axis=-1)
        T_diff = T_final - T_init
        std_T_diff = (np.std(temp[:10], axis=-1) + np.std(temp[-10:], axis=-1)) / 2
        density_ratios.append(ratio)
        temperature.append(T_diff)
        std_temperature.append(std_T_diff)

    temperature = np.array(temperature)
    density_ratios = np.array(density_ratios)
    std_temperature = np.array(std_temperature)

    plt.style.use(MPLSTYLE_FILE)
    plt.figure(figsize=(FIGURE_HALF_SIZE[0],2.5))
    plt.errorbar(density_ratios, temperature, yerr=std_temperature,
                ls="", lw=1.5, color="white", ecolor="black",
                marker="o", markersize=8, markeredgecolor="black", markeredgewidth=1.5)
    plt.gca().set(
        xscale="log",
        xlabel=r"Density ratio $n_\alpha\,/\,n_\text{p}$ (1)",
        ylabel=f"Heating $\\Delta T_{{{species.symbol()},x}}$ (eV)",
        xlim=(1e-2, 1e0),
    )
    if save:
        _saveFigure(f"heating-vs-density_ratio-{species.value}", "density_ratio_variation")

def wavenumberVsDensityRatio(save: bool=False):
    density_ratios = []
    k = []
    k_err = []
    for filename in sorted(DENSITY_VARIATION_FOLDER.glob("*.h5")):
        ratio = 10 ** float(filename.stem[-5:])
        if ratio >= 1e-1: continue
        density_ratios.append(ratio)
        info = runInfoForDenistyRatio(ratio)
        with h5py.File(filename) as f:
            time = f["Header/time"][:]
            grid = np.squeeze(f["Grid/grid"])[0]
            e_field = f["Electric Field/ex"][:]

        # fix units of time and energy
        time *= info.omega_pp
        grid /= info.lambda_D
        # extract particle numbers
        res = analysis.fitGrowthRate(
            time,
            np.mean(e_field ** 2, axis=1) * constants.epsilon_0 / (2.0 * constants.electron_volt),
            allowed_slope_deviation=0.5
        )
        if res is None:
            print(ratio)
        _, regime, _ = res
        print(regime[-1])
        k_single, k_err_single = analysis.estimateFrequency(
            -1, grid, e_field[regime[0]-100:regime[0]],
        )
        k.append(k_single) # res[0].slope/2
        k_err.append(k_err_single)

    plt.style.use(MPLSTYLE_FILE)
    plt.figure(figsize=FIGURE_HALF_SIZE)
    plt.errorbar(density_ratios, k, label="Simulation", # yerr=k_err,
        ls="", lw=1.5, marker="o", color="white", ecolor="black",
        markeredgecolor="black", markeredgewidth=1.5,
    )

    with h5py.File(THEORY_DENSITY_RATIO_FILE) as f:
        theory_k_max = f['k_max'][:]
        theory_ratio = f['na_np_ratio'][:]
    theory_k_max = np.array([k * runInfoForDenistyRatio(ratio).lambda_D for k, ratio in zip(theory_k_max, theory_ratio)])
    plt.plot(theory_ratio, theory_k_max, label="Theory")

    plt.xscale("log")
    plt.xlabel(r"Density ratio $n_\alpha\,/\,n_\text{p}$ (1)")
    plt.ylabel(r"Wave number $k_\text{max}\,\lambda_\text{D}$ (1)")
    plt.legend()
    if save:
        _saveFigure(f"wavenumber-vs-density_ratio", "density_ratio_variation")

def frequencyVsDensityRatio(save: bool=False):
    density_ratios = []
    omega = []
    omega_err = []
    for filename in sorted(DENSITY_VARIATION_FOLDER.glob("*.h5")):
        ratio = 10 ** float(filename.stem[-5:])
        if ratio >= 1e-1: continue
        density_ratios.append(ratio)
        info = runInfoForDenistyRatio(ratio)
        with h5py.File(filename) as f:
            time = f["Header/time"][:]
            grid = np.squeeze(f["Grid/grid"])
            e_field = f["Electric Field/ex"][:]

        # fix units of time and energy
        time *= info.omega_pp
        grid /= info.lambda_D
        # extract particle numbers
        _, regime, _ = analysis.fitGrowthRate(
            time,
            np.mean(e_field ** 2, axis=1) * constants.epsilon_0 / (2.0 * constants.electron_volt),
            allowed_slope_deviation=0.5
        )
        print(time[regime[-1]], filename.stem[-5:])
        E_field = e_field[:regime[-1]]
        omega_single, omega_err_single = analysis.estimateFrequency(
            -2, time, E_field,
        )
        f, p = signal.periodogram(E_field, axis=-2, fs=1/(time[1] - time[0]))
        p_mean = np.mean(p, axis=1)
        f *= 2 * np.pi
        def lorentzian( x, x0, gam, a, b):
            return a * gam**2 / ( gam**2 + ( x - x0 )**2) + b

        popt, pcov = optimize.curve_fit(
            lambda x, x0, gam, a, b: np.log(lorentzian(x, x0, gam, a, b)),
            f[f<np.pi][1:], np.log(p_mean[f<np.pi][1:]),
            p0=[f[1:][np.argmax(p_mean[1:])], 0.05, np.max(p_mean[1:]), 0])

        omega.append(omega_single)
        omega_err.append(omega_err_single)

    plt.style.use(MPLSTYLE_FILE)
    plt.figure(figsize=FIGURE_HALF_SIZE)
    plt.errorbar(density_ratios, omega, yerr=omega_err, label="Simulation",
        ls="", lw=1.5, marker="o", color="white", ecolor="black",
        markeredgecolor="black", markeredgewidth=1.5,
    )

    with h5py.File(THEORY_DENSITY_RATIO_FILE) as f:
        theory_omega_max = f['omega_max'][:]
        theory_ratio = f['na_np_ratio'][:]
    theory_omega_max = np.array([w / runInfoForDenistyRatio(ratio).omega_pp for w, ratio in zip(theory_omega_max, theory_ratio)])
    plt.plot(theory_ratio, theory_omega_max, label="Theory")

    plt.xscale("log")
    plt.xlabel(r"Density ratio $n_\alpha\,/\,n_\text{p}$ (1)")
    plt.ylabel(r"Wave frequency $\omega_\text{max}\,/\,\omega_\text{pp}$ (1)")
    plt.legend()
    if save:
        _saveFigure(f"frequency-vs-density_ratio", "density_ratio_variation")

def electricFieldDensityRatio(filename, save: bool=False):
    ratio = 10 ** float(filename.stem[-5:])
    info = runInfoForDenistyRatio(ratio)
    with h5py.File(filename) as f:
        time = f["Header/time"][:] * info.omega_pp
        grid_edges = np.squeeze(f["Grid/grid"][:])
        ex = f["Electric Field/ex"][:]

    grid_edges = grid_edges[0] / info.lambda_D

    plt.style.use(MPLSTYLE_FILE)
    plt.figure(figsize=(FIGURE_FULL_SIZE[0], 2.6))
    plt.pcolormesh(time, grid_edges, ex[:-1].T, cmap="bwr", rasterized=True,
                   vmin=-np.max(np.abs(ex)), vmax=np.max(np.abs(ex)))
    plt.colorbar(label=r"Electric field $E_x$ (V$\,/\,$m)")
    plt.xlabel("Time $t\\,\\omega_\\text{pp}$ (1)")
    plt.ylabel("Position $x\\,/\\,\\lambda_\\text{D}$ (1)")
    plt.gca().set(
        xlim=(0, 150),
        xticks=np.arange(0, 151, 30),
        ylim=(0, 64),
        yticks=np.arange(0, 65, 16),
    )
    if save:
        _saveFigure(f"electric_field-density_ratio_{filename.stem[-5:]}", "density_ratio_variation")

def linearRegimeDensityRatio(save: bool=False):
    plt.style.use(MPLSTYLE_FILE)
    plt.figure(figsize=FIGURE_HALF_SIZE)
    for filename in sorted(DENSITY_VARIATION_FOLDER.glob("density_*.h5")):
        ratio = 10 ** float(filename.stem.split("_")[-1])
        info = runInfoForDenistyRatio(ratio)
        with h5py.File(filename) as f:
            time = f["Header/time"][:] * info.omega_pp
            energy = f["Electric Field/ex"][:] ** 2
        energy = np.mean(
            energy,
            axis=tuple(range(1, energy.ndim))
        ) * (constants.epsilon_0 / 2) / (constants.electron_volt)
        fit_result = analysis.fitGrowthRate(time, energy, allowed_slope_deviation=0.1, reverse_search_direction=True)
        plt.plot(
            [ratio, ratio], time[fit_result[1]],
            marker="p", color="black", ls=":",
            markerfacecolor="white", markeredgecolor="black", markeredgewidth=1)
    plt.xscale("log")
    plt.xlim(1e-2, 1e0)
    plt.xlabel(r"Density ratio $n_\alpha\,/\,n_\text{p}$")
    plt.ylabel(r"Linear regime $t\,\omega_\text{pp}$ (1)")
    if save:
        _saveFigure(f"linear_regime-vs-density_ratio", "density_ratio_variation")

def growthRateDensityRatio(save: bool=False):
    density_ratios = []
    gamma = []
    for filename in sorted(DENSITY_VARIATION_FOLDER.glob("*.h5")):
        ratio = 10 ** float(filename.stem[-5:])
        density_ratios.append(ratio)
        info = runInfoForDenistyRatio(ratio)
        with h5py.File(filename) as f:
            time = f["Header/time"][:]
            grid = np.squeeze(f["Grid/grid"])[0]
            e_field = f["Electric Field/ex"][:]
        # fix units of time and energy
        time *= info.omega_pp
        grid /= info.lambda_D
        # extract particle numbers
        res = analysis.fitGrowthRate(
            time,
            np.mean(e_field ** 2, axis=1) * constants.epsilon_0 / (2.0 * constants.electron_volt),
            allowed_slope_deviation=0.15
        )
        gamma.append(res[0].slope / 2)
    with h5py.File(THEORY_DENSITY_RATIO_FILE) as f:
            theory_gamma = f['gamma_max'][:]
            theory_ratio = f['na_np_ratio'][:]

    plt.style.use(MPLSTYLE_FILE)
    plt.figure(figsize=FIGURE_HALF_SIZE)
    plt.plot(
        density_ratios,
        gamma,
        marker="o", color="white", ls="",
        markeredgecolor="black", markeredgewidth=1,
        label="Simulation",
    )
    plt.plot(
        theory_ratio,
        [g / runInfoForDenistyRatio(ratio).omega_pp for g in theory_gamma],
        label="Linear theory"
    )
    plt.xscale("log")
    plt.gca().set(
        xlim=(1e-2, 1e0),
        ylim=(0.0, 0.16),
        yticks=np.arange(0.0, 0.17, 0.04),
        xlabel=r"Density ratio $n_\alpha\,/\,n_\text{p}$",
        ylabel=r"Growth rate $\gamma\,/\,\omega_\text{pp}$",
    )
    plt.legend()
    if save:
        _saveFigure(f"growth_rate-vs-density_ratio", "density_ratio_variation")

def energyFractionsDensityRatio(save: bool=False):
    plt.figure(figsize=(FIGURE_HALF_SIZE[0],2.5))
    for idx, filename in enumerate(sorted(DENSITY_VARIATION_FOLDER.glob("density_*.h5"))):
        ratio = 10 ** float(filename.stem.split("_")[-1])
        info = runInfoForDenistyRatio(ratio)
        with h5py.File(filename) as f:
            time = f["Header/time"][:] * info.omega_pp
            e_avg_e = np.mean(f[f"Derived/Average_Particle_Energy/Electrons"], axis=1)
            e_avg_p = np.mean(f[f"Derived/Average_Particle_Energy/Protons"], axis=1)
            e_avg_a = np.mean(f[f"Derived/Average_Particle_Energy/Alphas"], axis=1)
            energy = f["Electric Field/ex"][:] ** 2
        energy = np.mean(
            energy,
            axis=tuple(range(1, energy.ndim))
        ) * (constants.epsilon_0 / 2) / (64 * info.lambda_D) ** 3
        fit_result = analysis.fitGrowthRate(time, energy, allowed_slope_deviation=0.1, reverse_search_direction=True)
        last_idx = (fit_result[1][1] + 880)
        plt.plot(ratio, (e_avg_a[0] - e_avg_a[last_idx] - (e_avg_p[last_idx] - e_avg_p[0]) - (e_avg_e[last_idx] - e_avg_e[0])) / (e_avg_a[0] - e_avg_a[last_idx]), marker="o", markeredgecolor="black", markeredgewidth=1, label=r"$W_E\,/\,|\Delta W_\alpha|$" if idx == 0 else None, color="white")
        plt.plot(ratio, (e_avg_p[last_idx] - e_avg_p[0]) / (e_avg_a[0] - e_avg_a[last_idx]), marker="v", markeredgecolor="black", markeredgewidth=1, label=r"$W_\text{p}\,/\,|\Delta W_\alpha|$" if idx == 0 else None, color="cornflowerblue")
        plt.plot(ratio, (e_avg_e[last_idx] - e_avg_e[0]) / (e_avg_a[0] - e_avg_a[last_idx]), marker="p", markeredgecolor="black", markeredgewidth=1, label=r"$W_\text{e}\,/\,|\Delta W_\alpha|$" if idx == 0 else None, color="gray")
    plt.xscale("log")
    plt.legend()
    plt.xlabel(r"Density ratio $n_\alpha\,/\,n_\text{p}$")
    plt.ylabel(r"Energy fractions $W\,/\,|\Delta W_\alpha|$ (1)" )
    plt.xlim(1e-2, 1e0)
    if save:
        _saveFigure(f"energy_fractions-vs-density_ratio", "density_ratio_variation")
