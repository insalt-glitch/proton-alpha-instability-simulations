from __future__ import annotations
from dataclasses import dataclass
import math

import numpy as np
from scipy import constants, optimize

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
    def lambda_D_exact(self: RunInfo) -> float:
        import numpy as np
        return np.sqrt(
            constants.epsilon_0 / (constants.elementary_charge * np.sum(
            np.array([
                -1, +1, +2,
            ]) ** 2 
            * np.array([
                self.electron.number_density,
                self.proton.number_density,
                self.alpha.number_density,
            ])
            / np.array([
                self.electron.temperature,
                self.proton.temperature,
                self.alpha.temperature,
            ])
        )))

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

    def ionSoundSpeedElectronProton(self: SpeciesInfo) -> float:
        """Ion acoustic speed (m/s)
        """
        return math.sqrt(constants.electron_volt / self.proton.si_mass * (
            1 * self.electron.temperature +
            3 * self.proton.temperature
        ))

    @property
    def c_s(self: SpeciesInfo):
        """Two ion sound speed for k * lambda_D = 0"""
        gamma_e = 1
        T_e = constants.electron_volt * self.electron.temperature
        m_i = [self.proton.si_mass, self.alpha.si_mass]
        gamma_i = [3, 3]
        T_i = [self.proton.temperature, self.alpha.temperature]
        Z_i = [1, 2]
        k_lambda = 0
        
        m_i = np.asarray(m_i)
        gamma_i = np.asarray(gamma_i)
        T_i = constants.electron_volt * np.asarray(T_i)
        Z_i = np.asarray(Z_i)
        A_i = m_i / constants.atomic_mass
        tau_i = gamma_i * T_i / (A_i * T_e)
    
        def f(v):
            u_squared = constants.atomic_mass * v ** 2 / T_e
            return (
                (gamma_e / np.mean(Z_i))
                * np.mean(Z_i ** 2 / A_i / (u_squared - tau_i))
                - (1 + gamma_e * (k_lambda) ** 2)
            )
        return optimize.fsolve(f, 100_000)[0]

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