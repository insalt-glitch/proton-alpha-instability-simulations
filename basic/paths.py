from pathlib import Path

RESULTS_FOLDER = Path("/home/nilsm/simulation_data")
FOLDER_1D = RESULTS_FOLDER / "epoch_1D"
FOLDER_2D = RESULTS_FOLDER / "epoch_2D"
PARTICLE_VARIATION_FOLDER = FOLDER_1D / "particle_variation"
DENSITY_VARIATION_FOLDER = FOLDER_1D / "density_variation"

ANALYSIS_FOLDER = Path(__file__).parent.parent

FIGURES_FOLDER = ANALYSIS_FOLDER / "figures"
MPLSTYLE_FILE = ANALYSIS_FOLDER / "plot_style.mplstyle"
