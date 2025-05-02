from pathlib import Path
from itertools import repeat

import h5py
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from numpy.typing import NDArray
from IPython.display import HTML

import analysis
from basic.paths import MPLSTYLE_FILE
from basic import RunInfo, Species, Distribution
from .settings import (
    FIGURE_DPI,
    FIGURE_FORMAT,
    FIGURES_FOLDER,
    VIDEO_FORMAT,
)

def generalSaveFigure(fig_name: str, sub_folder: str|None=None) -> None:
    """Saves a figure with 'fig_name' to a given sub-folder. Folder depends on the file-format.

    Args:
        fig_name (str): Name of the figure
        sub_folder (str | None, optional): Name of the sub-folder(s). Defaults to None.
    """
    folder = FIGURES_FOLDER / f"{FIGURE_FORMAT.lower()}"
    if sub_folder is not None:
        folder = folder / sub_folder
    folder.mkdir(exist_ok=True, parents=True)
    plt.savefig(
        folder / f"{fig_name}.{FIGURE_FORMAT.lower()}",
        dpi=FIGURE_DPI, bbox_inches="tight"
    )
    plt.clf()
    plt.close()

def generalSaveVideo(ani: animation.FuncAnimation, vid_name: str, sub_folder: str|None=None) -> None:
    folder = FIGURES_FOLDER / f"{VIDEO_FORMAT.lower()}"
    if sub_folder is not None:
        folder = folder / sub_folder
    folder.mkdir(exist_ok=True, parents=True)
    ani.save(
        folder / f"{vid_name}.{VIDEO_FORMAT.lower()}",
        writer="ffmpeg", fps=30
    )
    plt.clf()
    plt.close()

def plotEnergyEFieldOverTime(
    time: NDArray,
    energy: NDArray,
    show_fit_details: bool,
    show_legend: bool=True
):
    fit_result = analysis.fitGrowthRate(time, energy)

    plt.plot(time, energy, label="$\\langle W_E\\rangle_\\mathbf{r}^\\text{sim}$",
             color="black", lw=2)
    if fit_result is not None:
        lin_fit, fit_interval, poly_info = fit_result
        plt.plot(
            time[slice(*fit_interval)], energy[slice(*fit_interval)],
            color="orange", lw=3, ls="solid", zorder=3,
            label="Linear regime" if show_legend else None
        )
        plt.plot(
            time, np.exp(lin_fit.slope * time + lin_fit.intercept),
            ls="--", color="royalblue", zorder=9, lw=1.5,
            label="$W_{E}\\propto\\exp(2\\gamma\\,t)$" if show_legend else None
        )
        if show_fit_details:
            poly, extrema, turn_p = poly_info
            plt.plot(
                time, np.exp(poly(time)), label="Polynomial fit" if show_legend else None,
                color="gray", lw=1.5, ls=":", zorder=2, alpha=0.8
            )
            plt.plot(
                extrema, np.exp(poly(extrema)), label="Turning points" if show_legend else None,
                color="gray", zorder=3, ls="", alpha=0.8,
                marker="o", markeredgecolor="black", markeredgewidth=1.5, markersize=10,
            )
            plt.plot(
                turn_p, np.exp(poly(turn_p)), label="Inflection point" if show_legend else None,
                color="white", zorder=3, ls="", marker="p",
                markersize=10, markeredgecolor="black", markeredgewidth=1.5,
            )
    plt.xlim(0, 150)
    plt.xticks(np.linspace(0.0, 150.0, num=6))
    plt.yscale("log")
    plt.xlabel("Time $t\\,\\omega_\\text{pp}$ (1)")
    plt.ylabel("Energy $\\langle W_E\\rangle_x$ (eV$\\,/\\,$m$^3$)") # \\mathbf{r}
    if show_legend:
        plt.legend()

def _loadSpaceMomDistribution(
    info: RunInfo,
    species: Species,
    filename: Path,
    dist: Distribution,
    time: float|int|range,
    normalized_velocity: bool,
):
    with h5py.File(filename) as f:
        assert f"Grid/{dist}" in f, f"Distribution '{dist}' not in '{filename.as_posix()}'"
        is_2d_simulation = "Grid/grid/Y" in f
        sim_time = f["Header/time"][:] * info.omega_pp
        if isinstance(time, range):
            t_idx = time
        else:
            t_idx = np.argmin(np.abs(sim_time - time))
        if "Grid/grid/X" in f:
            x_grid = f["Grid/grid/X"][:]
        else:
            x_grid = f["Grid/grid"][:]
        if is_2d_simulation:
            y_grid = f[f"Grid/grid/Y"][:]
        mom_grid = f[f"Grid/{dist}/{species}/{dist.momentum()}"]
        if mom_grid.ndim > 1:
            mom_grid = np.squeeze(mom_grid[t_idx])
        else:
            mom_grid = np.squeeze(mom_grid[:])
        raw_dist = f[f'dist_fn/{dist}/{species}'][t_idx]
    if is_2d_simulation:
        v, f_v = analysis.normalizeDistributionXPx2D(
            x_grid, y_grid, mom_grid, raw_dist, info[species]
        )
    else:
        v, f_v = analysis.normalizeDistributionXPx1D(
            x_grid, mom_grid, raw_dist, info[species]
        )
    if normalized_velocity:
        v /= info[species].v_thermal
    else:
        v *= 1e-3
    return v, f_v

def momentumDistributionComparison(
    info: RunInfo,
    species: Species,
    dist_type: Distribution,
    files: list[Path]|Path,
    times: list[float]|float|int,
    labels: list[str]|None=None,
    legend_title: str|None=None,
    legend_ncols: int=1,
    legend_loc: str="best",
    normalized_velocity: bool=True,
    save: bool=False,
):
    assert isinstance(files, Path) or not isinstance(times, list), "Can only have multiple filesname or multiple times"
    assert legend_ncols >= 1, "Need at least one column"
    if isinstance(times, float):
        if isinstance(files, Path):
            files = [files]
            if legend_title is None:
                legend_title = "Time $t\\,\\omega_{pp}$ (1)"
            if labels is None:
                labels = [times]
        else:
            assert len(files) > 0, "Need to select at least one time"
            assert legend_title is not None, "Need to provide title for files"
            assert labels is not None, "Have to provide labels for files"
        assert len(labels) == len(files), "Need labels for each file"
        times = repeat(times)
    else:
        assert len(times) > 0, "Need to select at least one time"
        if labels is None:
            labels = times
        else:
            assert len(labels) == len(times), "Either use auto labels or provide one for each time."
        files = repeat(files)
        if legend_title is None:
            legend_title = "Time $t\\,\\omega_{pp}$ (1)"
    plt.style.use(MPLSTYLE_FILE)
    plt.figure()
    min_v, max_v = np.inf, - np.inf
    for t, filename, label in zip(times, files, labels):
        v, f_v = _loadSpaceMomDistribution(info, species, filename, dist_type, t, normalized_velocity)
        f_v = np.mean(f_v, axis=0)
        non_zero_v = v[np.nonzero(f_v > 0)]
        min_v = min(np.min(non_zero_v), min_v)
        max_v = max(np.max(non_zero_v), max_v)
        plt.plot(v, f_v, label=label)

    plt.yscale("log")
    if normalized_velocity:
        plt.xlabel(f"Velocity $v_{species.symbol()}\\,/\\,v^{{t=0}}_{species.symbol()}$ (1)")
    else:
        plt.xlabel(f"Velocity $v_{species.symbol()}$ (km/s)")
    plt.ylabel(f"$\\langle f_{species.symbol()}\\rangle_{{{dist_type.space().lower()}}}$ (s/m$^2$)")
    plt.legend(title=legend_title, ncols=legend_ncols, loc=legend_loc)
    plt.xlim(min_v, max_v)
    if save:
        generalSaveFigure(f"{dist_type.momentum().lower()}_distribution-{species}")

def spaceMomentumDistributon(
    info: RunInfo,
    species: Species,
    dist_type: Distribution,
    filename: Path,
    time: float|int,
    normalized_velocity: bool=True,
    save: bool=False,
):
    v, f_v = _loadSpaceMomDistribution(
        info, species, filename, dist_type, 60.0, normalized_velocity
    )
    dv = abs(v[1] - v[0])
    v = np.concat([[v[0]-dv], v]) + dv / 2

    with h5py.File(filename) as f:
        x_grid = f[f"Grid/grid/{dist_type.space()}"][:] / info.lambda_D
    non_zero_v = v[np.nonzero(np.sum(f_v, axis=0)>0)]
    f_v[f_v<=0] = np.min(f_v[f_v>0])

    plt.style.use(MPLSTYLE_FILE)
    plt.figure(figsize=(5.9, 3))
    plt.pcolormesh(x_grid, v, f_v.T, norm="log")
    plt.colorbar(label=f"$\\langle f_{species.symbol()}\\rangle_{{{dist_type.space()}}}$ (s/m$^2$)")
    plt.xlabel(f"Position {dist_type.space().lower()}$\\,/\\,\\lambda_\\text{{D}}$ (1)")
    if normalized_velocity:
        plt.ylabel(f"Velocity $v_{species.symbol()}\\,/\\,v^{{t=0}}_{species.symbol()}$ (1)")
    else:
        plt.ylabel(f"Velocity $v_{species.symbol()}$ (km/s)")
    plt.ylim(np.min(non_zero_v), np.max(non_zero_v))
    if save:
        generalSaveFigure(f"{dist_type}_distribution--{species}")

def videoMomentumDistribution(
    info: RunInfo,
    dist_type: Distribution,
    species: Species,
    filenames: Path|list[Path],
    time_steps: range|None=None,
    normalized_velocity: bool=True,
    legend_title: str|None=None,
    legend_ncols: int=1,
    labels: list[str]|None=None,
    save: bool=False
):
    if isinstance(filenames, list):
        if len(filenames) > 1:
            assert labels is not None, "Labels needed for multiple files"
            assert len(labels) == len(filenames), "Need one label per filename"
    else:
        filenames = [filenames]

    vs = []
    f_vs = []
    v_limits = []
    for file in filenames:
        with h5py.File(file) as f:
            time = f["Header/time"][:] * info.omega_pp
            if time_steps is None:
                time_steps = range(0, time.size, time.size // (30 * 5))
            time = time[time_steps]
        v, f_v = _loadSpaceMomDistribution(
            info, species, file, dist_type, time_steps, normalized_velocity,
        )
        f_v = np.mean(f_v, axis=1)
        non_zero_v = v[np.nonzero(np.sum(f_v, axis=0)>0)]
        vs.append(v)
        f_vs.append(f_v)
        v_limits.append([np.min(non_zero_v), np.max(non_zero_v)])

    plt.style.use(MPLSTYLE_FILE)
    fig, ax = plt.subplots()
    lines = []
    for idx, (v, f_v) in enumerate(zip(vs, f_vs)):
        if v.ndim > 1:
            line = ax.plot(v[0], f_v[0])[-1]
        else:
            line = ax.plot(v, f_v[0])[-1]
        if len(filenames) > 1:
            line.set_label(labels[idx])
        lines.append(line)
    text = ax.text(0.95, 0.95,
            horizontalalignment='right',
            verticalalignment='top',
            s=f"t$\\,\\omega_\\text{{pp}}=\\,${time[0]:>5.1f}",
            transform=ax.transAxes
    )
    plt.yscale("log")
    if normalized_velocity:
        plt.xlabel(f"Velocity $v_{species.symbol()}\\,/\\,v^{{t=0}}_{species.symbol()}$ (1)")
    else:
        plt.xlabel(f"Velocity $v_{species.symbol()}$ (km/s)")
    plt.ylabel(f"$\\langle f_{species.symbol()}\\rangle_{{{dist_type.space()}}}$ (s/m$^2$)")
    plt.xlim(np.min(v_limits), np.max(v_limits))
    if len(filenames) > 1:
        plt.legend(loc="lower center", title=legend_title, ncols=legend_ncols)

    tight_bbox = fig.get_tightbbox()
    fig.set_size_inches(tight_bbox.width, tight_bbox.height)
    fig.set_layout_engine("tight", pad=0.01)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    def update(frame_idx):
        for line, v, f_v in zip(lines, vs, f_vs):
            if v.ndim > 1:
                line.set_xdata(v[frame_idx])
            line.set_ydata(f_v[frame_idx])
        text.set_text(f"t$\\,\\omega_{{pp}}\\,=\\,${time[frame_idx]:>5.1f}")
        return text, *lines

    ani = animation.FuncAnimation(fig=fig, func=update, frames=range(len(list(time_steps))))
    if save:
        generalSaveVideo(
            ani,
            f"{dist_type.momentum().lower()}_distribution-{species}",
            f"{dist_type.momentum().lower()}_distribution"
        )
    else:
        return HTML(ani.to_jshtml())
