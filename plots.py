from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants

import analysis
import physics

from definitions import (
    RESULTS_FOLDER,
    MPLSTYLE_FILE,
    FIGURES_FOLDER,
    Species,
    SpeciesInfo,
    RunInfo
)

FIGURE_FORMAT = "svg"
FIGURE_DPI = 200


def electricFieldOverSpaceAndTime(
    folder: Path,
    info: RunInfo,
    save: bool=False
):
    time, (grid_edges, ex), _ = analysis.readFromRun(
        folder,
        ["Grid/grid", "Electric Field/ex"]
    )
    grid_edges = grid_edges[0] / info.lambda_D
    time *= info.omega_pp

    plt.style.use(MPLSTYLE_FILE)
    plt.figure(figsize=(6,3.5))
    plt.pcolormesh(time, grid_edges, ex[:-1].T, cmap="bwr")
    plt.colorbar(label="Electric field E (V/m)")
    plt.xlabel("Time $t\\,\\omega_{pp}$ (1)")
    plt.ylabel("Position n / $\\lambda_D$ (1)")
    plt.yticks(np.arange(5) * 32)
    plt.ylim(0, np.max(grid_edges))
    plt.tight_layout()
    if save:
        FIGURES_FOLDER.mkdir(exist_ok=True)
        plt.savefig(
            FIGURES_FOLDER / f"e_field-vs-time-vs-space.{FIGURE_FORMAT}",
            dpi=FIGURE_DPI, bbox_inches="tight"
        )
        plt.clf()

def averageTemperature3DOverTime(
    folder: Path,
    info: RunInfo,
    save: bool=False
):

    time, (T_electron, T_proton, T_alpha), _ = analysis.readFromRun(
        folder = folder,
        dataset_names=[f"Derived/Temperature/{species.value}" for species in Species],
        processElement=lambda x: physics.kelvinToElectronVolt(np.mean(x))
    )
    time *= info.omega_pp

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
        FIGURES_FOLDER.mkdir(exist_ok=True)
        plt.savefig(
            FIGURES_FOLDER / f"avg_temp3d-vs-time.{FIGURE_FORMAT}",
            dpi=FIGURE_DPI, bbox_inches="tight"
        )
        plt.clf()

def velocityDistributionOverTime(
    folder: Path,
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
    time, (dist,), _ = analysis.readFromRun(
        folder=folder, processElement=lambda x: np.mean(x, axis=0),
        dataset_names=[f"/dist_fn/x_px/{species.value}"]
    )
    time *= info.omega_pp

    _, (grid,), _ = analysis.readFromRun(
        folder=folder, time_interval=0,
        dataset_names=[f"Grid/x_px/{species.value}/Px"],
    )
    plt.style.use(MPLSTYLE_FILE)
    species_info = info[species]
    normalized_grid = grid / species_info.p_thermal
    dx = 0.5 * info.lambda_D
    dv = (grid[1] - grid[0]) / (species_info.mass * constants.electron_mass)
    dist = dist.T / (dx * dv)
    dist[dist<=0] = np.min(dist[dist>0])

    fig = plt.figure(figsize=(8,4))
    quad = plt.pcolormesh(time, normalized_grid, dist, norm="log")
    plt.colorbar(label=f"$f_{LABEL[species]}$ (s/m$^2$)")
    plt.xlabel("Time $t\\,\\omega_{pp}$ (1)")
    plt.ylabel(f"Velocity $v_{LABEL[species]}\\,/\\,v_{{\\text{{thermal}},{LABEL[species]}}}$")
    plt.ylim(*Y_LIM[species])
    plt.tight_layout()
    if save:
        FIGURES_FOLDER.mkdir(exist_ok=True)
        plt.savefig(
            FIGURES_FOLDER / f"velocity_dist-vs-time_{species.value}.{FIGURE_FORMAT}",
            dpi=FIGURE_DPI, bbox_inches="tight"
        )
        plt.clf()

def electricFieldEnergyOverTime(
    folder: Path,
    info: RunInfo,
    show_fit: bool=True,
    save: bool=False
):
    time, (energy,), _ = analysis.readFromRun(
        folder=folder, processElement=lambda x: np.mean(np.array(x) ** 2),
        dataset_names=["/Electric Field/ex"]
    )
    # convert types and normalize
    time *= info.omega_pp
    energy *= (constants.epsilon_0 / 2) / (constants.electron_volt)
    fit_result = analysis.fitGrowthRate(time, energy)

    plt.style.use(MPLSTYLE_FILE)
    plt.figure()
    plt.plot(time, energy, alpha=0.7, label="Simulation data")
    if show_fit and fit_result is not None:
        lin_fit, fit_interval, poly_info = fit_result
        poly, extrema, turn_p = poly_info
        plt.plot(extrema, np.exp(poly(extrema)),
                color="lightblue",zorder=10, ls="", alpha=0.8,
                marker="o", markeredgecolor="black", markeredgewidth=1, markersize=10,
                label="Extrema")
        plt.plot(turn_p, np.exp(poly(turn_p)),
                color="red", zorder=10, ls="", marker="p",
                markersize=10, markeredgecolor="black", markeredgewidth=1,
                label="Turning point")
        plt.plot(time[slice(*fit_interval)], energy[slice(*fit_interval)],
                color="blue", lw=2, ls="solid", zorder=3, alpha=0.6,
                label="Linear regime")
        plt.plot(time, np.exp(lin_fit.slope * time + lin_fit.intercept),
                ls=":", color="black", zorder=9,
                label="$W_{E}\\propto\\exp(2\\gamma\\,t)$")
        plt.plot(time, np.exp(poly(time)),
            color="black", lw=1, ls="-.", zorder=2, alpha=0.8,
            label="Polynomial fit")
    plt.xlim(0, 150)
    plt.ylim(1e4, 1e7)
    plt.xticks(np.linspace(0.0, 150.0, num=6))
    plt.yscale("log")
    plt.xlabel("Time $t\\,\\omega_{pp}$ (1)")
    plt.ylabel("Energy $W_E$ (eV$\\,/\\,$m$^3$)")
    plt.legend()
    if save:
        FIGURES_FOLDER.mkdir(exist_ok=True)
        plt.savefig(
            FIGURES_FOLDER / "e_field_energy-vs-time.{FIGURE_FORMAT}",
            dpi=FIGURE_DPI, bbox_inches="tight"
        )
        plt.clf()

def multiElectricFieldEnergyOverTime(
    folder: Path,
    info: RunInfo,
    identifer: str|None=None,
    save: bool=False
):
    time, (energies,), folders = analysis.readFromRun(
        folder=folder,
        dataset_names=["/Electric Field/ex"],
        processElement=lambda x: np.mean(np.array(x) ** 2),
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
            poly, extrema, turn_p = poly_info
            plt.plot(extrema, np.exp(poly(extrema)),
                    color="lightblue", zorder=10, ls="", alpha=0.8,
                    marker="o", markeredgecolor="black", markeredgewidth=1, markersize=10,
                    label="Extrema" if is_first_fit else None)
            plt.plot(turn_p, np.exp(poly(turn_p)),
                    color="red", zorder=10, ls="", marker="p",
                    markersize=10, markeredgecolor="black", markeredgewidth=1,
                    label="Turning point" if is_first_fit else None)
            plt.plot(time[slice(*fit_interval)], W_E[slice(*fit_interval)],
                    color="blue", lw=2, ls="solid", zorder=3, alpha=0.6,
                    label="Linear regime" if is_first_fit else None)
            plt.plot(time, np.exp(lin_fit.slope * time + lin_fit.intercept),
                    ls=":", color="black", zorder=9,
                    label="$W_{E}\\propto\\exp(2\\gamma\\,t)$" if is_first_fit else None)
            plt.plot(time, np.exp(poly(time)),
                    color="black", lw=1, ls="-.", zorder=2, alpha=0.8,
                    label= "Polynomial fit" if is_first_fit else None)
            is_first_fit = False
        plt.plot(time, W_E, alpha=0.6, label=f"Run {run_idx}", zorder=1)

    plt.xlim(0, 150)
    plt.ylim(4e4, 2e7)
    plt.yscale("log")
    plt.xlabel("Time $t\\,\\omega_{pp}$ (1)")
    plt.ylabel("Energy $W_E$ (eV$\\,/\\,$m$^3$)")
    plt.tight_layout()
    plt.legend(
        ncols=2, loc="lower right",
        fancybox=False, labelspacing=0.4, columnspacing=1
    )
    if save:
        FIGURES_FOLDER.mkdir(exist_ok=True)
        plt.savefig(
            FIGURES_FOLDER / f"e_field_energy-vs-time_{identifer}.{FIGURE_FORMAT}",
            dpi=FIGURE_DPI, bbox_inches="tight"
        )
        plt.clf()
    else:
        plt.title(identifer)

def particleVariationGrowthRate(
    info: RunInfo,
    save: bool=False
):
    time, (energies,), folders = analysis.readFromRun(
        folder=RESULTS_FOLDER / "particle_variation",
        dataset_names=["/Electric Field/ex"],
        processElement=lambda x: np.mean(np.array(x) ** 2),
        recursive=True
    )
    # fix units of time and energy
    time *= info.omega_pp
    energies *= constants.epsilon_0 / (2.0 * constants.electron_volt)
    # extract particle numbers
    particle_numbers = np.array([int(pfs[0][0].parent.stem[-4:]) for pfs in folders])
    # extract growth rates from fits
    fits = [[analysis.fitGrowthRate(time, W_E) for W_E in es] for es in energies]

    growth_rates = [[res[0].slope / 2 for res in fs if res is not None] for fs in fits]
    growth_rates_mean = np.array([np.mean(x) if len(x) == 4 else np.nan for x in growth_rates])
    growth_rates_std  = np.array([np.std(x)  if len(x) == 4 else np.nan for x in growth_rates])

    plt.errorbar(
        particle_numbers, growth_rates_mean, yerr=growth_rates_std, ls="",
        marker="p", markersize=8, markeredgecolor="black", markeredgewidth=1
    )
    # TODO: This smeels and should be fixed
    plt.plot(
        2 ** 13, 8.4622E-02, ls="",
        marker="p", markersize=8, markeredgecolor="black", markeredgewidth=1
    )
    plt.fill_between([0, particle_numbers[np.isnan(growth_rates_mean)].max()], [0.0, 0.0], [2.0,2.0], color="red", alpha=0.6)
    plt.xscale("log", base=2)
    plt.xlim(0.5 * particle_numbers[0], 2 ** 14)
    plt.ylim(0.0, 1e-1)
    plt.yticks(np.linspace(0.0, 1e-1, 6))
    plt.xlabel("Simulated particles per cell")
    plt.ylabel("Growth rate $\\gamma\\,/\\,\\omega_{pp}$ (1)")
    if save:
        FIGURES_FOLDER.mkdir(exist_ok=True)
        plt.savefig(
            FIGURES_FOLDER / f"growth_rate-vs-num_particles.{FIGURE_FORMAT}",
            dpi=FIGURE_DPI, bbox_inches="tight"
        )
        plt.clf()

def particleVariationTemperature3D(
    species: Species,
    save: bool=False
):
    time, (temperatures,), folders = analysis.readFromRun(
        folder=RESULTS_FOLDER / "particle_variation",
        dataset_names=[f"Derived/Temperature/{species.value}"],
        processElement=lambda x: physics.kelvinToElectronVolt(np.mean(x)),
        recursive=True
    )
    # extract particle numbers
    particle_numbers = np.array([int(pfs[0][0].parent.stem[-4:]) for pfs in folders])
    # compute quantities of interest
    temperatures = np.mean(temperatures[:,:,-20:], axis=-1)
    mean_T = np.mean(temperatures, axis=1)
    std_T = np.std(temperatures, axis=1)
    plt.style.use(MPLSTYLE_FILE)
    plt.figure()
    plt.errorbar(particle_numbers, mean_T, yerr=std_T, ls="",
                    marker="p", markersize=10, markeredgecolor="black", markeredgewidth=1)
    plt.xscale("log", base=2)
    plt.xlabel("Simulated particles per cell")
    plt.ylabel("Temperature $T_{final}$ (eV)")
    if save:
        FIGURES_FOLDER.mkdir(exist_ok=True)
        plt.savefig(
            FIGURES_FOLDER / f"{species.value}_temperature_3D-vs-num_particles.{FIGURE_FORMAT}",
            dpi=FIGURE_DPI, bbox_inches="tight"
        )
        plt.clf()
    else:
        plt.title(species)

def particleVariationEnergyVsTime(
    info: RunInfo,
    save: bool=False
):
    variation_folder = RESULTS_FOLDER / "particle_variation"
    particle_numbers = []
    time, (energies,), folders = analysis.readFromRun(
        folder=RESULTS_FOLDER / "particle_variation",
        dataset_names=["/Electric Field/ex"],
        processElement=lambda x: np.mean(np.array(x) ** 2),
        recursive=True
    )
    # fix units of time and energy
    time *= info.omega_pp
    energies *= constants.epsilon_0 / (2.0 * constants.electron_volt)
    energies = np.cumsum(np.mean(energies, axis=1), axis=-1)
    energies = (energies[:,10:] - energies[:,:-10]) / 10
    # extract particle numbers
    particle_numbers = np.array([int(pfs[0][0].parent.stem[-4:]) for pfs in folders])
    plt.style.use("plot_style.mplstyle")
    plt.figure()
    for num_p, W_E in zip(particle_numbers, energies):
        plt.plot(time[5:-5], W_E, label=num_p)
    plt.yscale("log")
    plt.xlabel("Time $t\\,\\omega_{pp}$ (1)")
    plt.ylabel("Energy $W_E$ (eV$\\,/\\,$m$^3$)")
    plt.legend(title="Simulated particles $N_\\text{sim}\\,/\\,N_c$", ncols=2)
    plt.xlim(time[0], time[-1])
    plt.xticks(np.linspace(0, 150, 6))
    if save:
        FIGURES_FOLDER.mkdir(exist_ok=True)
        plt.savefig(
            FIGURES_FOLDER / f"avg_e_field_energy-vs-time-vs-num_particles.{FIGURE_FORMAT}",
            dpi=FIGURE_DPI, bbox_inches="tight"
        )
        plt.clf()

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
    variation_folder = RESULTS_FOLDER / "particle_variation"
    time, (dist,), folders = analysis.readFromRun(
        folder = RESULTS_FOLDER / "particle_variation",
        dataset_names=[f"/dist_fn/x_px/{species.value}"],
        processElement=lambda x: np.mean(x, axis=0),
        recursive=True
    )
    _, (grid,), _ = analysis.readFromRun(
        folder = RESULTS_FOLDER / "particle_variation/particles_0032/rep_0",
        dataset_names=[f"Grid/x_px/{species.value}/Px"],
        time_interval=0
    )
    time *= info.omega_pp
    # extract particle numbers
    particle_numbers = np.array([int(pfs[0][0].parent.stem[-4:]) for pfs in folders])

    temperature = analysis.avgTemperatureFromMomentumDist(grid, dist, info, species)
    T_init = np.mean(temperature[:,:,:10], axis=-1)
    T_final = np.mean(temperature[:,:,-10:], axis=-1)
    T_diff = T_final - T_init
    mean_T_diff = np.mean(T_diff, axis=-1)
    std_T_diff = np.std(T_diff, axis=-1)

    plt.style.use(MPLSTYLE_FILE)
    plt.figure()
    plt.errorbar(particle_numbers, mean_T_diff, yerr=std_T_diff, label="Simulation",
                ls="", marker="p", markersize=10, markeredgecolor="black", markeredgewidth=1)
    plt.axhline(EXPECTED_T_DIFF[species], color="black", ls="--", label="Graham")
    plt.xscale("log", base=2)
    plt.xlabel("Simulated particles $N_\\text{sim}\\,/\\,N_c$")
    plt.ylabel("Temperature $\\Delta T_x$ (eV)")
    plt.legend()
    if save:
        FIGURES_FOLDER.mkdir(exist_ok=True)
        plt.savefig(
            FIGURES_FOLDER / f"{species.value}_temperature_diff-vs-num_particles.{FIGURE_FORMAT}",
            dpi=FIGURE_DPI, bbox_inches="tight"
        )
        plt.clf()
    else:
        plt.title(species)


def particleVariationTemperatureXVsTime(
    info: RunInfo,
    species: Species,
    save: bool=False
):
    expected_diffs = [102 - 100, 14.5 - 3, 50 - 12]
    variation_folder = RESULTS_FOLDER / "particle_variation"
    time, (dist,), folders = analysis.readFromRun(
        folder = RESULTS_FOLDER / "particle_variation",
        dataset_names=[f"/dist_fn/x_px/{species.value}"],
        processElement=lambda x: np.mean(x, axis=0),
        recursive=True
    )
    _, (grid,), _ = analysis.readFromRun(
        folder = RESULTS_FOLDER / "particle_variation/particles_0032/rep_0",
        dataset_names=[f"Grid/x_px/{species.value}/Px"],
        time_interval=0
    )
    time *= info.omega_pp
    # extract particle numbers
    particle_numbers = np.array([int(pfs[0][0].parent.stem[-4:]) for pfs in folders])

    temperature = analysis.avgTemperatureFromMomentumDist(grid, dist, info, species)

    plt.style.use(MPLSTYLE_FILE)
    plt.figure()
    plt.plot(time, np.mean(temperature[2:], axis=1).T, label=particle_numbers[2:])
    plt.xlabel("Time $t\\,\\omega_{pp}$ (1)")
    plt.ylabel("Temperature $T_x$ (eV)")
    plt.legend(title="Simulated particles $N_\\text{sim}\\,/\\,N_c$ (1)", ncols=2)
    if save:
        FIGURES_FOLDER.mkdir(exist_ok=True)
        plt.savefig(
            FIGURES_FOLDER / f"{species.value}_temperature_x-vs-time-vs-num_particles.{FIGURE_FORMAT}",
            dpi=FIGURE_DPI, bbox_inches="tight"
        )
        plt.clf()
    else:
        plt.title(species)

def particleVariationWavenumber(
    info: RunInfo,
    save: bool=False
):
    time, (electric_fields,), folders = analysis.readFromRun(
        folder=RESULTS_FOLDER / "particle_variation",
        dataset_names=["/Electric Field/ex"],
        recursive=True
    )
    _, (grid,), _ = analysis.readFromRun(
        folder=RESULTS_FOLDER / "particle_variation/particles_0032/rep_0",
        dataset_names=["/Grid/grid"],
        time_interval=0,
    )
    # fix units of time and energy
    time *= info.omega_pp
    energies = np.mean(electric_fields ** 2, axis=-1)
    energies *= constants.epsilon_0 / (2.0 * constants.electron_volt)
    dx = (grid[1] - grid[0]) / info.lambda_D
    N = electric_fields.shape[-1]
    # extract particle numbers
    particle_numbers = np.array([int(pfs[0][0].parent.stem[-4:]) for pfs in folders])
    # extract linear regimes from fits
    fits = [[analysis.fitGrowthRate(time, W_E) for W_E in es] for es in energies]
    fit_success_idx = np.array([all(fit is not None for fit in fs) for fs in fits])
    electric_fields = electric_fields[fit_success_idx]
    regimes = [[slice(*res[1]) for res in fs] for idx, fs in enumerate(fits) if fit_success_idx[idx]]

    k = np.empty(electric_fields.shape[:2])
    k_err = np.empty(electric_fields.shape[:2])
    particle_numbers = np.array(particle_numbers)
    for p_idx, (rs, es) in enumerate(zip(regimes, electric_fields)):
        for rep_idx, (linear_regime, e_field) in enumerate(zip(rs, es)):
            e_field_linear = e_field[linear_regime]
            e_field_fft = np.abs(np.fft.rfft(e_field_linear, axis=1)) ** 2
            k[p_idx,rep_idx] = np.mean(2 * np.pi * np.argmax(e_field_fft, axis=1) / (dx * N))
            k_err[p_idx,rep_idx] = np.mean(4 * np.pi / (dx * N * np.sqrt(2)))

    mean_k = np.mean(k, axis=1)
    mean_k_err = np.mean(k_err, axis=1)
    plt.errorbar(particle_numbers[fit_success_idx], mean_k, yerr=mean_k_err, label="Simulation",
                ls="", marker="p", markersize=10, markeredgecolor="black", markeredgewidth=1)
    plt.fill_between(
        [0, particle_numbers[~fit_success_idx].max()],
        [0.0, 0.0],
        [2.0, 2.0],
        color="red", alpha=0.6
    )
    plt.xscale("log", base=2)
    plt.xlim(0.5 * particle_numbers[0], 2 ** 14)
    plt.ylim(0.2, 0.8)
    plt.axhline(0.96 * info.lambda_D / info.lambda_D_electron, color="black", ls=":", label="Graham")
    plt.yticks(np.linspace(0.2, 0.8, num=4))
    plt.xlabel("Simulated particles $N_\\text{sim}\\,/\\,N_c$ (1)")
    plt.ylabel("Wave numbers $k\\,\\lambda_{D}$ (1)")
    plt.legend()
    if save:
        FIGURES_FOLDER.mkdir(exist_ok=True)
        plt.savefig(
            FIGURES_FOLDER / f"wavenumber-vs-num_particles.{FIGURE_FORMAT}",
            dpi=FIGURE_DPI, bbox_inches="tight"
        )
        plt.clf()

def particleVariationFrequency(
    info: RunInfo,
    save: bool=False
):
    time, (electric_fields,), folders = analysis.readFromRun(
        folder=RESULTS_FOLDER / "particle_variation",
        dataset_names=["/Electric Field/ex"],
        recursive=True
    )
    _, (grid,), _ = analysis.readFromRun(
        folder=RESULTS_FOLDER / "particle_variation/particles_0032/rep_0",
        dataset_names=["/Grid/grid"],
        time_interval=0,
    )
    # fix units of time and energy
    time *= info.omega_pp
    energies = np.mean(electric_fields ** 2, axis=-1)
    energies *= constants.epsilon_0 / (2.0 * constants.electron_volt)
    dt = (time[1] - time[0])
    # extract particle numbers
    particle_numbers = np.array([int(pfs[0][0].parent.stem[-4:]) for pfs in folders])
    # extract linear regimes from fits
    fits = [[analysis.fitGrowthRate(time, W_E) for W_E in es] for es in energies]
    fit_success_idx = np.array([all(fit is not None for fit in fs) for fs in fits])
    electric_fields = electric_fields[fit_success_idx]
    regimes = [[slice(*res[1]) for res in fs] for idx, fs in enumerate(fits) if fit_success_idx[idx]]

    omega = np.empty(electric_fields.shape[:2])
    omega_err = np.empty(electric_fields.shape[:2])
    particle_numbers = np.array(particle_numbers)
    for p_idx, (rs, es) in enumerate(zip(regimes, electric_fields)):
        for rep_idx, (linear_regime, e_field) in enumerate(zip(rs, es)):
            linear_regime = slice(
                2 * linear_regime.start - linear_regime.stop,
                linear_regime.stop
            )
            e_field_linear = e_field[linear_regime]
            N = e_field_linear.shape[0]
            e_field_fft = np.abs(np.fft.rfft(e_field_linear, axis=0)) ** 2
            omega[p_idx,rep_idx] = np.mean(2 * np.pi * np.argmax(e_field_fft, axis=0) / (dt * N))
            omega_err[p_idx,rep_idx] = np.mean(4 * np.pi / (dt * N * np.sqrt(2)))

    mean_omega = np.mean(omega, axis=1)
    mean_omega_err = np.mean(omega_err, axis=1)
    plt.errorbar(particle_numbers[fit_success_idx][1:], mean_omega[1:], yerr=mean_omega_err[1:], label="Simulation",
                ls="", marker="p", markersize=10, markeredgecolor="black", markeredgewidth=1)
    plt.plot(2 ** 13, 0.66, ls="", marker="p", markersize=10, markeredgecolor="black", markeredgewidth=1)
    plt.fill_between(
        [0, particle_numbers[~fit_success_idx].max() * 2],
        [0.0, 0.0],
        [3.0, 3.0],
        color="red", alpha=0.6
    )
    plt.xscale("log", base=2)
    plt.xlim(0.5 * particle_numbers[0], 2 ** 14)
    plt.ylim(0.0, 3.0)
    plt.axhline(0.72, color="black", ls=":", label="Graham")
    plt.xlabel("Simulated particles $N_\\text{sim}\\,/\\,N_c$ (1)")
    plt.ylabel("Freqency $\\omega\\,/\\,\\omega_{pp}$ (1)")
    plt.legend()
    if save:
        FIGURES_FOLDER.mkdir(exist_ok=True)
        plt.savefig(
            FIGURES_FOLDER / f"frequency-vs-num_particles.{FIGURE_FORMAT}",
            dpi=FIGURE_DPI, bbox_inches="tight"
        )
        plt.clf()
