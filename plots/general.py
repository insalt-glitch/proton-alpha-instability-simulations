from pathlib import Path
from itertools import repeat

import h5py
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
    VIDEO_DPI,
    FIGURE_FULL_SIZE,
    FIGURE_HALF_SIZE,
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
        dpi=VIDEO_DPI,
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
    x_lim: tuple[float,float]|None=None,
    y_lim: tuple[float|None,float|None]=(None,None),
    x_ticks=None,
    normalized_velocity: bool=True,
    save: bool=False,
    save_folder: str|None=None,
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
            legend_title = r"Time $t\,\omega_\text{pp}$ (1)"
    plt.style.use(MPLSTYLE_FILE)
    plt.figure(figsize=FIGURE_HALF_SIZE)
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
        plt.xlabel(rf"Velocity $v_{species.symbol()}\,/\,v^{{t=0}}_{{\text{{t}}{species.symbol()}}}$ (1)")
    else:
        plt.xlabel(f"Velocity $v_{species.symbol()}$ (km$\\,/\\$s)")
    plt.ylabel(f"Distribution $\\langle f_{species.symbol()}\\rangle_{{{dist_type.space().lower()}}}$ (s$\\,/\\,$m$^2$)")
    plt.legend(title=legend_title, ncols=legend_ncols, loc=legend_loc, labelspacing=0.2)
    if x_lim is None:
        x_lim = (min_v, max_v)
    plt.xlim(x_lim)
    if x_ticks is not None:
        plt.xticks(x_ticks)
    plt.ylim(y_lim)

    if save:
        generalSaveFigure(f"{dist_type.momentum().lower()}_distribution-{species}", save_folder)

def spaceMomentumDistributon(
    info: RunInfo,
    species: Species,
    dist_type: Distribution,
    filename: Path,
    time: float|int,
    normalized_velocity: bool=True,
    v_lim: tuple[float|None,float|None]|None=None,
    save: bool=False,
):
    v, f_v = _loadSpaceMomDistribution(
        info, species, filename, dist_type, time, normalized_velocity
    )
    dv = abs(v[1] - v[0])
    v = np.concat([[v[0]-dv], v]) + dv / 2

    with h5py.File(filename) as f:
        if f"Grid/grid/{dist_type.space()}" in f:
            x_grid = f[f"Grid/grid/{dist_type.space()}"][:] / info.lambda_D
        else:
            x_grid = f["Grid/grid"][:] / info.lambda_D

    plt.style.use(MPLSTYLE_FILE)
    plt.figure(figsize=(FIGURE_FULL_SIZE[0], 1.5))
    plt.pcolormesh(x_grid, v, f_v.T, norm="log", cmap=plt.cm.get_cmap("viridis"))
    cax = plt.colorbar(label=f"$\\langle f_{species.symbol()}\\rangle_{{{dist_type.space()}}}$ (s$\\,/\\,$m$^2$)")
    plt.gca().set_facecolor(cax.cmap.get_under())
    plt.xlabel(f"Position {dist_type.space().lower()}$\\,/\\,\\lambda_\\text{{D}}$ (1)")
    if normalized_velocity:
        plt.ylabel(f"Velocity $v_{species.symbol()}\\,/\\,v^{{t=0}}_{species.symbol()}$ (1)")
    else:
        plt.ylabel(f"Velocity $v_{species.symbol()}$ (km$\\,/\\,$s)")
    if v_lim is None:
        v_lim = (np.min(v), np.max(v))
    plt.ylim(v_lim)
    plt.xticks(np.linspace(0, 128, 5))
    if save:
        generalSaveFigure(f"{dist_type}_distribution--{species}")

def spaceVelocityDistributionMulti(
    info: RunInfo,
    species: Species,
    dist_type: Distribution,
    filename: Path,
    times: tuple[float|int,float|int,float|int],
    normalized_velocity: bool=True,
    v_lim: tuple[float|None,float|None]|None=None,
    v_ticks: list[float]|None=None,
    subfig_offset: int=0,
    save: bool=False,
    save_folder: str|None=None,
):
    assert subfig_offset>=0, "Start at (a) == 0 or higher, must be non-negative"
    v_list = []
    f_v_list = []
    max_f = -np.inf
    min_f = np.inf
    for t in times:
        v, f_v = _loadSpaceMomDistribution(
            info, species, filename, dist_type, t, normalized_velocity
        )
        v_list.append(v)
        f_v_list.append(f_v)
        max_f = max(np.max(f_v), max_f)
        min_f = min(np.min(f_v[f_v>0]), min_f)

    plt.style.use(MPLSTYLE_FILE)
    fig, axes = plt.subplots(
        3, 1, figsize=(FIGURE_FULL_SIZE[0], 3.3),
        sharex=True, height_ratios=(1,1,1),
    )
    axes: list[plt.Axes] = axes
    ax_idx = subfig_offset
    for t, ax, v, f_v in zip(times, axes, v_list, f_v_list):
        dv = abs(v[1] - v[0])
        v = np.concat([[v[0]-dv], v]) + dv / 2

        with h5py.File(filename) as f:
            if f"Grid/grid/{dist_type.space()}" in f:
                x_grid = np.squeeze(f[f"Grid/grid/{dist_type.space().lower()}"][:]) / info.lambda_D
            else:
                x_grid = np.squeeze(f["Grid/grid"][:]) / info.lambda_D
        quad = ax.pcolormesh(
            x_grid, v, f_v.T / max_f, norm="log",
            cmap=plt.cm.get_cmap("viridis"),
            vmin=1e-2, vmax=1.0, rasterized=True, # min_f/max_f
        )

        if v_lim is None:
            v_lim = (np.min(v), np.max(v))
        ax.set(
            ylim=v_lim,
            facecolor=plt.cm.get_cmap("viridis").get_under(),
        )
        ax.text(
            0.01, 0.95, s=f"$\\mathbf{{({chr(ord('a')+ax_idx)})\\,\\,t\\,\\omega_\\text{{pp}}={t}}}$",
            horizontalalignment='left',
            verticalalignment='top',
            color="white",
            transform=ax.transAxes,
        )
        ax_idx += 1
        if v_ticks is not None: ax.set_yticks(v_ticks)
    axes[-1].set(
        xlabel=f"Position {dist_type.space().lower()}$\\,/\\,\\lambda_\\text{{D}}$ (1)",
        xticks=np.linspace(0, 128, 9),
    )
    if normalized_velocity:
        y_label = f"$v_{species.symbol()}\\,/\\,v^{{t=0}}_{species.symbol()}$ (1)"
    else:
        y_label = f"$v_{species.symbol()}$ (km$\\,/\\,$s)"
    axes[1].set_ylabel(f"Velocity {y_label}")
    fig.colorbar(
        quad,
        ax=axes.ravel().tolist(),
        orientation="vertical",
        use_gridspec=True,
        # fraction=0.03,
        pad=0.02,
        label=f"Distribution $f_{species.symbol()}$ (s$\\,/\\,$m$^2$)",
    )
    # plt.tight_layout(pad=0.1)
    if save:
        generalSaveFigure(f"space-velocity-dist-multi-timestep_{species}", save_folder)

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
    plt.ylabel(f"$\\langle f_{species.symbol()}\\rangle_{{{dist_type.space()}}}$ (s$\\,/\\$m$^2$)")
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

def videoEvolutionDistributionFunction(
    info: RunInfo, species: Species, filename: Path,
    dist_type: Distribution, time: range, vlim: tuple=None, vticks: list=None, save: bool=False):
    v, f_v = _loadSpaceMomDistribution(
        info, species, filename, dist_type, time, True
    )
    dv = abs(v[:,1] - v[:,0])
    v = np.concat([(v[:,0]-dv)[:,None], v], axis=1) + dv[:,None] / 2
    with h5py.File(filename) as f:
        if f"Grid/grid/{dist_type.space()}" in f:
            x_grid = f[f"Grid/grid/{dist_type.space()}"][:] / info.lambda_D
        else:
            x_grid = np.squeeze(f["Grid/grid"][:]) / info.lambda_D
    f_v /= np.max(f_v[0])
    plt.style.use(MPLSTYLE_FILE)
    fig, ax = plt.subplots(figsize=(FIGURE_FULL_SIZE[0], 2))
    quad = plt.pcolormesh(x_grid, v[0], f_v[0].T, norm="log", cmap=plt.get_cmap("viridis"))
    cax = plt.colorbar(label=f"$\\langle f_{species.symbol()}\\rangle_{{{dist_type.space()}}}$ (a.u.)")
    text = plt.text(
        0.98, 0.95, s=rf"$t\,\omega_\text{{pp}}={time[0] / 10}$",
        horizontalalignment='right',
        verticalalignment='top',
        color="white",
        transform=ax.transAxes,
    )
    ax.set_facecolor(cax.cmap.get_under())
    plt.xlabel(f"Position {dist_type.space().lower()}$\\,/\\,\\lambda_\\text{{D}}$ (1)")
    plt.ylabel(f"Velocity $v_{species.symbol()}\\,/\\,v^{{t=0}}_{{\\text{{t}}{species.symbol()}}}$ (1)")
    if vlim is None:
        plt.ylim(np.min(v), np.max(v))
    else:
        plt.ylim(*vlim)
    if vticks:
        plt.yticks(vticks)
    plt.xticks(np.linspace(0, 128, 5))
    tight_bbox = fig.get_tightbbox()
    fig.set_size_inches(tight_bbox.width, tight_bbox.height)
    fig.set_layout_engine("tight", pad=0.1)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    def update(frame_idx):
        quad.set_array(f_v[frame_idx].T)
        quad._coordinates = np.array(np.meshgrid(v[frame_idx], x_grid)[::-1]).T
        text.set_text(f"t$\\,\\omega_\\text{{pp}}\\,=\\,${time[frame_idx]//10:>5.1f}")
        return (quad, text,)
    frames = list(range(len(list(time))))
    ani = animation.FuncAnimation(fig=fig, func=update, frames=frames)
    if save:
        generalSaveVideo(
            ani,
            f"x_px_distribution-{species}",
            "x_px_distribution"
        )
    else:
        return HTML(ani.to_jshtml())