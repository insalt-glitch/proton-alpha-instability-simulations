import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .settings import (
    FIGURE_DPI,
    FIGURE_FORMAT,
    FIGURES_FOLDER
)
from basic.paths import MPLSTYLE_FILE
import analysis

def generalSaveFigure(fig_name: str, sub_folder: str|None = None) -> None:
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

def plotEnergyEFieldOverTime(
    time: NDArray,
    energy: NDArray,
    show_fit_details: bool,
    show_legend: bool=True
):
    fit_result = analysis.fitGrowthRate(time, energy)

    plt.plot(time, energy, alpha=0.7, label="$\\langle W_E\\rangle_\\mathbf{r}^\\text{sim}$")
    if fit_result is not None:
        lin_fit, fit_interval, poly_info = fit_result
        plt.plot(
            time[slice(*fit_interval)], energy[slice(*fit_interval)],
            color="blue", lw=2, ls="solid", zorder=3, alpha=0.6,
            label="Linear regime" if show_legend else None
        )
        plt.plot(
            time, np.exp(lin_fit.slope * time + lin_fit.intercept),
            ls=":", color="black", zorder=9,
            label="$W_{E}\\propto\\exp(2\\gamma\\,t)$" if show_legend else None
        )
        if show_fit_details:
            poly, extrema, turn_p = poly_info
            plt.plot(
                time, np.exp(poly(time)), label="Polynomial fit" if show_legend else None,
                color="black", lw=1, ls="-.", zorder=2, alpha=0.8,
            )
            plt.plot(
                extrema, np.exp(poly(extrema)), label="Extrema" if show_legend else None,
                color="lightblue",zorder=10, ls="", alpha=0.8,
                marker="o", markeredgecolor="black", markeredgewidth=1, markersize=10,
            )
            plt.plot(
                turn_p, np.exp(poly(turn_p)), label="Turning point" if show_legend else None,
                color="red", zorder=10, ls="", marker="p",
                markersize=10, markeredgecolor="black", markeredgewidth=1,
            )
    plt.xlim(0, 150)
    plt.xticks(np.linspace(0.0, 150.0, num=6))
    plt.yscale("log")
    plt.xlabel("Time $t\\,\\omega_{pp}$ (1)")
    plt.ylabel("Energy $\\langle W_E\\rangle_\\mathbf{r}$ (eV$\\,/\\,$m$^3$)")
    if show_legend:
        plt.legend()
