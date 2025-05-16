import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import constants as const

def electronVoltToKelvin(temperature: ArrayLike) -> NDArray | float:
    """Conversion: eV -> K

    Args:
        temperature (ArrayLike): Temperature (eV)

    Returns:
        NDArray | float: Temperature (K)
    """
    return temperature * const.elementary_charge / const.k

def kelvinToElectronVolt(temperature: ArrayLike) -> NDArray | float:
    """Conversion: K -> eV

    Args:
        temperature (ArrayLike): Temperature (K)

    Returns:
        NDArray | float: Temperature (eV)
    """
    return temperature * const.k / const.elementary_charge

def temperatureToThermalSpeed(temperature: ArrayLike, mass: ArrayLike) -> NDArray | float:
    """Convert temperature (eV) to the equivalent thermal speed

    Args:
        temperature (ArrayLike): Temperature (eV)
        mass (ArrayLike): Mass (electron-mass)

    Returns:
        NDArray | float: Thermal speed (m/s)
    """
    return np.sqrt(
        2.0 * temperature * const.elementary_charge
        / (mass * const.electron_mass)
    )

def plasmaFrequency(mass: ArrayLike, charge: ArrayLike, number_density: ArrayLike) -> NDArray | float:
    """Calculate the plasma frequency of a given species.

    Args:
        mass (ArrayLike): Mass (electron-mass)
        charge (ArrayLike): Charge (electron-charge)
        density (ArrayLike): Number density (m^-3)

    Returns:
        NDArray | float: Plasma frequency (Hz)
    """
    return np.sqrt(
        (charge * const.elementary_charge) ** 2 * number_density
        / (const.epsilon_0 * mass * const.electron_mass)
    )

def debyeLength(
    electron_temperature: ArrayLike,
    densities: ArrayLike,
    charges: ArrayLike
) -> NDArray | float:
    """Computes the Debye length in meters

    Args:
        electron_temperature (ArrayLike): Electron temperature (eV).
        densities (ArrayLike): Number densities of all species (m^-3)
        charges (ArrayLike): Charges of all species (elementary-charge)

    Returns:
        NDArray | float: Debye length (m)
    """
    electron_temperature = np.array(electron_temperature)
    densities = np.array(densities)
    charges = np.array(charges)
    return np.sqrt(
        const.epsilon_0 * electron_temperature  /
        np.sum(const.elementary_charge * densities * charges ** 2)
    )
