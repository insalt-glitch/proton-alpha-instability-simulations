import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy import io
from typing import Callable

from basic import RunInfo
from basic.paths import THEORY_DISPERSION_FOLDER

def _densityRatio(dispersion_data: dict[str, NDArray]) -> NDArray:
    return np.squeeze(dispersion_data['navec'] / dispersion_data['npvec'])

def _growthRate(dispersion_data: dict[str, NDArray]) -> NDArray:
    return np.squeeze(dispersion_data['gammamax'] / dispersion_data['wppvec'])

def _waveVector(dispersion_data: dict[str, NDArray], info: RunInfo) -> NDArray:
    return np.squeeze(dispersion_data['kmax'] * info.lambda_D)

def _waveFrequency(dispersion_data: dict[str, NDArray]) -> NDArray:
    return np.squeeze(dispersion_data['wmax'] / dispersion_data['wppvec'])

def _metaLoadAndInterpolate(
    n_alpha_over_n_proton: ArrayLike,
    func: Callable,
    info: RunInfo|None=None
) -> ArrayLike:
    dispersion_data = io.loadmat(
        THEORY_DISPERSION_FOLDER / "alpha_proton_density_ratio.mat"
    )['disprelnanprat'][0,0]

    density_ratio = _densityRatio(dispersion_data)
    if info is not None:
        quantity = func(dispersion_data, info)
    else:
        quantity = func(dispersion_data)
    return np.interp(n_alpha_over_n_proton, density_ratio, quantity)

def growthRate(n_alpha_over_n_proton: ArrayLike) -> ArrayLike:
    return _metaLoadAndInterpolate(n_alpha_over_n_proton, _growthRate)

def waveNumber(n_alpha_over_n_proton: ArrayLike, info: RunInfo) -> ArrayLike:
    return _metaLoadAndInterpolate(n_alpha_over_n_proton, _waveVector, info)

def waveFrequency(n_alpha_over_n_proton: ArrayLike) -> ArrayLike:
    return _metaLoadAndInterpolate(n_alpha_over_n_proton, _waveFrequency)
