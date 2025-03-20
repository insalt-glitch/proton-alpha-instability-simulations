from basic.paths import FIGURES_FOLDER
import matplotlib.pyplot as plt

FIGURE_FORMAT = "svg"
FIGURE_DPI = 200
FIGURE_FULL_SIZE = (5.9, 4.2)
FIGURE_LINESTYLES = ['solid', (0, (1, 1)), (0, (5, 2)), (0, (5, 1, 1, 1, 1, 1)), (0, (5, 1, 1, 1)), (0, (2,1,2,1,2,3)), (0, (1, 1, 1, 3)), (3,(8,1,8,1)), (0,(5,1,1,1,5,1,1,1,1,1)), (3,1,3,1,1,1,1,1)]

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
