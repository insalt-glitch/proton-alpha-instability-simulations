from pathlib import Path

import colormaps as cmaps
import matplotlib.legend_handler
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy import constants, signal
import h5py

import analysis
import theory
from basic import physics
from .general import generalSaveFigure, plotEnergyEFieldOverTime

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
    plt.xlabel("Time $t\\,\\omega_{pp}$ (1)")
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
        plt.figure(figsize=(FIGURE_HALF_SIZE[0]-0.3, 2.5))
    else:
        plt.figure(figsize=(FIGURE_HALF_SIZE[0], 2.5))
    plt.plot(time, temperature, label=f"$T_{{{species.symbol()},x}}$")
    plt.plot(time, temp_3d, label=f"$T_{{{species.symbol()},\\text{{3D}}}}$")
    plt.xlabel("Time $t\\,\\omega_{pp}$ (1)")
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
    filename: Path, info: RunInfo, save: bool=False
):
    plt.style.use(MPLSTYLE_FILE)
    fig, axes = plt.subplots(3, 1, figsize=(FIGURE_FULL_SIZE[0], 7), sharex=True)
    Y_LIMS = [(-4, 4), (-4, 8), (-1, 8)]
    for ax, species, y_lim in zip(axes, Species, Y_LIMS):
        with h5py.File(PARTICLE_VARIATION_FOLDER / "particles_8192/rep_0.h5") as f:
            time = f["Header/time"][:] * info.omega_pp
            x_grid = f[f"Grid/x_px/{species.value}/X"][:]
            px_grid = f[f"Grid/x_px/{species.value}/Px"][:]
            dist = np.mean(f[f"/dist_fn/x_px/{species.value}"][:], axis=1)

        v, f_v = analysis.normalizeDistributionXPx1D(
            x_grid, px_grid, dist, info[species]
        )
        relative_v = v / info[species].v_thermal
        if relative_v.ndim > 1:
            time = np.tile(time, (px_grid.shape[1], 1)).T
            f_v = f_v.T
        
        quad = ax.pcolormesh(time, relative_v, f_v.T, norm="log", rasterized=True)
        ax.set(
            ylabel = (f"Velocity $v_{species.symbol()}\\,/\\,v^{{t=0}}_{{\\text{{th}},{species.symbol()}}}$ (1)"),
            xlim = (0,150),
            xticks = (np.linspace(0, 150, 6)),
            ylim=y_lim,
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cax = plt.colorbar(quad, label=f"$\\langle f_{species.symbol()}\\rangle_x$ (s$\\,/\\,$m$^2$)", cax=cax)
        ax.set_facecolor(cax.cmap.get_under())
    ax.set(xlabel="Time $t\\,\\omega_\\text{pp}$ (1)")
    plt.tight_layout()
    if save:
        _saveFigure(f"velocity_dist-vs-time")

def energyEFieldOverTime(
    filename: Path,
    info: RunInfo,
    show_fit_details: bool=True,
    save: bool=False
):
    with h5py.File(filename) as f:
        time = f["Header/time"][:] * info.omega_pp
        energy = np.mean(f["Electric Field/ex"][:] ** 2, axis=1) * (constants.epsilon_0 / 2) / (constants.electron_volt)

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
    plt.xlabel("Time $t\\,\\omega_{pp}$ (1)")
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
    plt.xlabel("Time $t\\,\\omega_{pp}$ (1)")
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
