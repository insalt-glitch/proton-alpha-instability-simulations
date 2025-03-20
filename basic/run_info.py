from __future__ import annotations
from dataclasses import dataclass
import math

from scipy import constants

from . import physics, SpeciesInfo, Species

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

    @property
    def c_s(self: SpeciesInfo) -> float:
        """Ion acoustic speed (m/s)
        """
        si_proton_mass = self.proton.mass * constants.electron_mass
        return math.sqrt(constants.electron_volt / si_proton_mass * (
            1 * self.electron.temperature +
            3 * self.proton.temperature
        ))

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