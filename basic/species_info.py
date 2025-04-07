from __future__ import annotations
from dataclasses import dataclass

from scipy import constants

from . import physics

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
        return self.si_mass * self.v_thermal

    @property
    def si_mass(self: SpeciesInfo) -> float:
        """Mass (kg)
        """
        return self.mass * constants.electron_mass

    @property
    def si_temperature(self: SpeciesInfo):
        """Temperature (K)
        """
        return physics.electronVoltToKelvin(self.temperature)

    @property
    def si_charge(self: SpeciesInfo):
        return self.charge * constants.elementary_charge
