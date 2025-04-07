from pathlib import Path
import matplotlib.pyplot as plt
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
    plt.figure(figsize=(6,3.5))
    plt.pcolormesh(time, grid_edges, ex[:-1].T, cmap="bwr", rasterized=True,
                   vmin=-np.max(np.abs(ex)), vmax=np.max(np.abs(ex)))
    plt.colorbar(label="Electric field E (V/m)")
    plt.xlabel("Time $t\\,\\omega_{pp}$ (1)")
    plt.ylabel("Position n / $\\lambda_D$ (1)")
    plt.yticks(np.arange(5) * 32)
    plt.ylim(0, np.max(grid_edges))
    plt.xlim(0, 150)
    plt.tight_layout()
    if save:
        _saveFigure("e_field-vs-time-vs-space")

def averageTemperature3DOverTime(
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

def velocityDistributionOverTime(
    filename: Path,
    info: RunInfo,
    species: Species,
    save: bool=False
):
    Y_LIM = {
        Species.ELECTRON: (-5,  5),
        Species.PROTON  : (-4, 10),
        Species.ALPHA   : (-3, 10)
    }
    LABEL = {
        Species.ELECTRON: "e",
        Species.PROTON  : "p",
        Species.ALPHA   : "\\alpha",
    }
    with h5py.File(filename) as f:
        time = f["Header/time"][:] * info.omega_pp
        x_grid = f[f"Grid/x_px/{species.value}/X"][:]
        px_grid = f[f"Grid/x_px/{species.value}/Px"][:]
        dist = np.mean(f[f"/dist_fn/x_px/{species.value}"][:], axis=1)

    plt.style.use(MPLSTYLE_FILE)
    v, f_v = analysis.normalizeDistributionXPx1D(
        x_grid, px_grid, dist, info[species]
    )
    relative_v = v / info[species].v_thermal
    f_v[f_v<=0] = np.min(f_v[f_v>0])

    fig = plt.figure(figsize=(8,4))
    quad = plt.pcolormesh(time, relative_v, f_v.T, norm="log", rasterized=True)
    plt.colorbar(label=f"$f_{LABEL[species]}$ (s/m$^2$)")
    plt.xlabel("Time $t\\,\\omega_{pp}$ (1)")
    plt.ylabel(f"Velocity $v_{LABEL[species]}\\,/\\,v_{{\\text{{thermal}},{LABEL[species]}}}$")
    plt.ylim(*Y_LIM[species])
    plt.tight_layout()
    if save:
        _saveFigure(f"velocity_dist-vs-time_{species.value}")

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
    plt.figure()
    plotEnergyEFieldOverTime(time, energy, show_fit_details=show_fit_details)
    plt.ylim(1e4, 1e7)
    if save:
        _saveFigure("e_field_energy-vs-time")

def multiElectricFieldEnergyOverTime(
    folder: Path,
    info: RunInfo,
    identifer: str|None=None,
    save: bool=False
):
    time, (energies,), _ = analysis.readFromVariation(
        folder=folder,
        dataset_names=["/Electric Field/ex"],
        processElement=lambda x: np.mean(np.array(x) ** 2, axis=1),
        recursive=True
    )
    # fix units of time and energy
    time *= info.omega_pp
    energies *= constants.epsilon_0 / (2.0 * constants.electron_volt)

    plt.style.use(MPLSTYLE_FILE)
    plt.figure()
    is_first_fit = True
    for run_idx, W_E in enumerate(energies):
        fit_result = analysis.fitGrowthRate(time, W_E)
        if fit_result is not None:
            lin_fit, fit_interval, poly_info = fit_result
            plt.plot(time[slice(*fit_interval)], W_E[slice(*fit_interval)],
                    color="blue", lw=2, ls="solid", zorder=3, alpha=0.6,
                    label="Linear regime" if is_first_fit else None)
            plt.plot(time, np.exp(lin_fit.slope * time + lin_fit.intercept),
                    ls=":", color="black", zorder=9,
                    label="$W_{E}\\propto\\exp(2\\gamma\\,t)$" if is_first_fit else None)
            is_first_fit = False
        plt.plot(time, W_E, alpha=0.6, label=f"Run {run_idx}", zorder=1)

    plt.xlim(0, 150)
    plt.ylim(1e4, 2e7)
    plt.yscale("log")
    plt.xlabel("Time $t\\,\\omega_{pp}$ (1)")
    plt.ylabel("Energy $W_E$ (eV$\\,/\\,$m$^3$)")
    plt.tight_layout()
    plt.legend(
        ncols=2, loc="lower right",
        fancybox=False, labelspacing=0.4, columnspacing=1
    )
    if save:
        _saveFigure(f"e_field_energy-vs-time_{identifer}", "particles_per_cell")
    else:
        plt.title(identifer)

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

    plt.errorbar(
        particle_numbers, growth_rates_mean, yerr=growth_rates_std, ls="", label="Simulation",
        marker="p", markersize=8, markeredgecolor="black", markeredgewidth=1,
    )
    plt.plot(
        8192, p8192_growth_rate, ls="",
        marker="p", markersize=8, markeredgecolor="black", markeredgewidth=1
    )
    plt.axhline(theory.growthRate(1e-1), label="Graham", color="black", ls=":")
    y_min, y_max = plt.gca().get_ylim()
    plt.fill_between(
        [0, particle_numbers[np.isnan(growth_rates_mean)].max()],
        y1=y_min, y2=y_max,
        color="red", alpha=0.6, label="Failed")
    plt.xscale("log", base=2)
    plt.xlim(0.5 * particle_numbers[0], 2 ** 14)
    plt.ylim(y_min, y_max)
    plt.yticks(np.linspace(0.2e-1, 1.2e-1, 6))
    plt.xlabel("Simulated particles per cell")
    plt.ylabel("Growth rate $\\gamma\\,/\\,\\omega_{pp}$ (1)")
    plt.legend(loc="lower right")
    if save:
        _saveFigure("growth_rate-vs-num_particles", "particles_per_cell")

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
    print(temperatures)
    mean_T = np.mean(temperatures, axis=1)
    std_T = np.std(temperatures, axis=1)
    p8192_T = np.mean(p8192_temperature[-20:])

    plt.style.use(MPLSTYLE_FILE)
    plt.figure()
    plt.errorbar(particle_numbers, mean_T, yerr=std_T, ls="",
        marker="p", markersize=8, markeredgecolor="black", markeredgewidth=1)
    plt.plot(8192, p8192_T, ls="",
        marker="p", markersize=8, markeredgecolor="black", markeredgewidth=1)
    plt.xscale("log", base=2)
    plt.xlabel("Simulated particles per cell")
    plt.ylabel("Temperature $T_{final}$ (eV)")
    plt.xlim(2 ** 4, 2 ** 14)
    if save:
        _saveFigure(f"temperature_3D-vs-num_particles_{species.value}", "particles_per_cell")
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
    plt.style.use("plot_style.mplstyle")
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
    EXPECTED_T_DIFF = {
        Species.ELECTRON: 102 - 100,
        Species.PROTON: 14.5 - 3,
        Species.ALPHA: 50 - 12
    }
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
    plt.figure()
    plt.errorbar(particle_numbers, mean_T_diff, yerr=std_T_diff, label="Simulation",
                ls="", marker="p", markersize=8, markeredgecolor="black", markeredgewidth=1)
    plt.plot(8192, p8192_T_diff, ls="",
        marker="p", markersize=8, markeredgecolor="black", markeredgewidth=1)
    plt.axhline(EXPECTED_T_DIFF[species], color="black", ls="--", label="Graham")
    plt.xscale("log", base=2)
    plt.xlabel("Simulated particles $N_\\text{sim}\\,/\\,N_c$")
    plt.ylabel("Temperature $\\Delta T_x$ (eV)")
    plt.legend(loc="center right")
    if save:
        _saveFigure(f"temperature_diff-vs-num_particles_{species.value}", "particles_per_cell")
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
        _saveFigure(f"temperature_x-vs-time-vs-num_particles_{species.value}", "particles_per_cell")
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
    p8192_k, p8192_k_err = analysis.estimateFrequency(-1, p8192_grid, p8192_E_field[...,:350,:], peak_cutoff=0.9)

    mean_k = np.mean(k, axis=1)
    mean_k_err = np.std(k, axis=1)
    mean_k_err[mean_k_err < k_err] = k_err
    plt.errorbar(particle_numbers[2:], mean_k[2:], yerr=mean_k_err[2:],
        label="Simulation", ls="", marker="p",
        markersize=10, markeredgecolor="black", markeredgewidth=1)
    plt.errorbar(8192, p8192_k, yerr=p8192_k_err, ls="",
        marker="p", markersize=8, markeredgecolor="black", markeredgewidth=1)
    plt.axhline(theory.waveNumber(1e-1, info), color="black", ls=":", label="Graham")
    y_min, y_max = plt.gca().get_ylim()
    plt.fill_between(
        [0, particle_numbers[1]],
        y1=y_min, y2=y_max,
        color="red", alpha=0.6, label="Failed"
    )
    plt.xscale("log", base=2)
    plt.xlim(0.5 * particle_numbers[0], 2 ** 14)
    plt.ylim(y_min, y_max)
    plt.yticks(np.linspace(0.2, 0.7, num=6))
    plt.xlabel("Simulated particles $N_\\text{sim}\\,/\\,N_c$ (1)")
    plt.ylabel("Wave numbers $k\\,\\lambda_{D}$ (1)")
    plt.legend(loc="upper left")
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

    omega, omega_err = analysis.estimateFrequency(-2, time, E_fields[...,:400,:])
    p8192_omega, p8192_omega_err = analysis.estimateFrequency(-2, p8192_time, p8192_E_field[...,:400,:])

    mean_omega = np.mean(omega, axis=1)
    mean_omega_err = np.std(omega, axis=1)
    mean_omega_err[mean_omega_err < omega_err] = omega_err
    plt.errorbar(particle_numbers[2:], mean_omega[2:], yerr=mean_omega_err[2:],
        label="Simulation", ls="", marker="p",
        markersize=10, markeredgecolor="black", markeredgewidth=1)
    plt.errorbar(2 ** 13, p8192_omega, yerr=p8192_omega_err,
        ls="", marker="p",
        markersize=10, markeredgecolor="black", markeredgewidth=1)

    plt.axhline(theory.waveFrequency(1e-1), color="black", ls=":", label="Graham")
    y_min, y_max = plt.gca().get_ylim()
    plt.fill_between(
        [0, particle_numbers[1]],
        y1=0.5, y2=y_max,
        color="red", alpha=0.6, label="Failed"
    )
    plt.xscale("log", base=2)
    plt.xlim(0.5 * particle_numbers[0], 2 ** 14)
    plt.ylim(0.5, y_max)
    plt.yticks(np.linspace(0.5, 0.8, num=4))
    plt.xlabel("Simulated particles $N_\\text{sim}\\,/\\,N_c$ (1)")
    plt.ylabel("Freqency $\\omega\\,/\\,\\omega_{pp}$ (1)")
    plt.legend(loc="upper left")
    if save:
        _saveFigure("frequency-vs-num_particles", "particles_per_cell")
