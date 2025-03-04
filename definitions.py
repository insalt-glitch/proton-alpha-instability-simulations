from __future__ import annotations
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
import physics
from scipy import constants

RESULTS_FOLDER = Path("/home/nilsm/simulation_data")
ANALYSIS_FOLDER = Path(__file__).parent

FIGURES_FOLDER = ANALYSIS_FOLDER / "figures"
MPLSTYLE_FILE = ANALYSIS_FOLDER / "plot_style.mplstyle"

class Species(Enum):
    ELECTRON = "Electrons"
    PROTON   = "Protons"
    ALPHA    = "Alphas"

    def __eq__(self, other):
        return self.value == other.value

    def __hash__(self):
        return self.value.__hash__()

@dataclass
class SpeciesInfo:
    """Basic information about a species

    Args:
        number_density (float): Number density (m^-3)
        temperature (float): Temperature (eV)
        charge (float): Electric charge (electron-charge)
        mass (float): Particle mass (electron-mass)
        bulk_velocity (float): Bulk velocity (m/s)
    """
    # Number density (m^-3)
    number_density: float
    temperature: float
    charge: float
    mass: float
    bulk_velocity: float

    @property
    def omega(self: SpeciesInfo) -> float:
        """Angular plasma frequency (Hz)
        """
        return physics.plasmaFrequency(self.mass, self.number_density)

    @property
    def v_thermal(self: SpeciesInfo) -> float:
        """Themal speed (m/s)
        """
        return physics.temperatureToThermalSpeed(
            self.temperature, self.mass
        )

    @property
    def p_thermal(self: SpeciesInfo) -> float:
        """Thermal momentum (kg*m/s)
        """
        si_mass = self.mass * constants.electron_mass
        return si_mass * self.v_thermal

@dataclass
class RunInfo:
    """Basic information about the species in a simulation run.
    """
    electron: SpeciesInfo
    proton: SpeciesInfo
    alpha: SpeciesInfo

    @property
    def lambda_D_electron(self: RunInfo) -> float:
        """Debye length (m) considering ONLY ELECTRONS
        """
        return physics.debyeLength(
            self.electron.temperature,
            densities=[self.electron.number_density],
            charges=[-1]
        )

    @property
    def lambda_D(self: RunInfo) -> float:
        """Debye length (m)
        """
        return physics.debyeLength(
            self.electron.temperature,
            densities=[
                self.electron.number_density,
                self.proton.number_density,
                self.alpha.number_density
            ],
            charges=[-1, +1, +2]
        )

    @property
    def omega_pe(self: RunInfo) -> float:
        """Angular electron plasma frequency (Hz)
        """
        return self.electron.omega

    @property
    def omega_pp(self: RunInfo) -> float:
        """Angular proton plasma frequency (Hz)
        """
        return self.proton.omega

    def __iter__(self: RunInfo):
        for species_info in [self.electron, self.proton, self.alpha]:
            yield species_info

    def __getitem__(self: RunInfo, key: Species):
        mapping = {
            Species.ELECTRON: self.electron,
            Species.PROTON: self.proton,
            Species.ALPHA: self.alpha
        }
        result = mapping.get(key, None)
        if result is None:
            raise RuntimeError("What did you do?")
        return result
