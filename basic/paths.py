from pathlib import Path
from . import Variation

RESULTS_FOLDER = Path("/mnt/internal_hdd/sim_data")

FOLDER_1D = RESULTS_FOLDER / "epoch_1D"
PARTICLE_VARIATION_FOLDER = FOLDER_1D / "particle_variation"
DENSITY_VARIATION_FOLDER = FOLDER_1D / "density_ratio_variation"
THEORY_DISPERSION_FOLDER = RESULTS_FOLDER / "linear_theory"

THEORY_DENSITY_RATIO_FILE = THEORY_DISPERSION_FOLDER / "theory_density_ratio.h5"
THEORY_U_ALPHA_FILE = THEORY_DISPERSION_FOLDER / "theory_u_alpha_dispersion.h5"

FOLDER_2D = RESULTS_FOLDER / "epoch_2D"
V_FLOW_VARIATION_FOLDER = FOLDER_2D / "v_alpha_bulk_variation"

ANALYSIS_FOLDER = Path(__file__).parent.parent

FIGURES_FOLDER = ANALYSIS_FOLDER / "figures"
MPLSTYLE_FILE = ANALYSIS_FOLDER / "plots/plot_style.mplstyle"

VARIATION_FILES = {
    Variation.U_ALPHA_FLOW: sorted(V_FLOW_VARIATION_FOLDER.glob("*.h5")),
}
