import numpy as np
from scipy import stats, constants
from scipy.stats._stats_py import LinregressResult
from numpy.typing import ArrayLike, NDArray
from numpy.polynomial import Polynomial
from collections.abc import Callable, Generator
from typing import Any
from pathlib import Path
import h5py

from definitions import RunInfo, Species

def fitGrowthRate(
    time: NDArray,
    field_energy: NDArray,
    polynomial_degree: int=7,
    min_interval_length: float=5.0,
    allowed_slope_deviation: float=0.2,
) -> tuple[
    LinregressResult,
    NDArray,
    tuple[Polynomial, NDArray, NDArray]
] | None:
    """Try to fit the growth rate from the energy of the electric field over time.

    The works such that a high-order polynomial is fitted to all data. From
    that we compute the extremum with lowest y-value. This extremum is assumed
    to left-adjacent to the linear regime. The turning-point next to it will be
    center of the fit-interval. The fit interval consists of time-values where
    the slope of high-order polynomial is similar to that of the turning-point.

    Args:
        time (NDArray): Time data (omega_pp)
        field_energy (NDArray): Energy of the electric field (eV/m^3)
        polynomial_degree (int, optional): Degree of the polynomial used for
            deterimining linear-regime. Defaults to 7.
        min_interval_length (float, optional): Minimum length of the
            linear-regime. Defaults to 5.0.
        allowed_slope_deviation (float, optional): Maximum allowed relative
            difference in slope compared to the turning-point. Defaults to 0.2.

    Returns:
        tuple[LinregressResult,NDArray,tuple[Polynomial,NDArray,NDArray]]|None: tuple of:
            1) scipy fit-result
            2) interval used to fit the data
            3) Information about the polyfit:
                3.1) fitted polynomial
                3.2) location of extrema
                3.3) location of turning-point which is the center of the fit-interval
    """
    assert time.size > 10, "Expect minimum length of array"
    assert allowed_slope_deviation > 0, "Slope difference cannot be negative"
    poly = Polynomial.fit(time[5:], np.log(field_energy[5:]), deg=polynomial_degree)
    # calculate first and second derivative
    dpoly_dt1 = poly.deriv()
    dpoly_dt2 = dpoly_dt1.deriv()
    # get extrema and turning points
    poly_extrema = np.real(dpoly_dt1.roots()[np.isreal(dpoly_dt1.roots())])
    poly_turn_p = np.real(dpoly_dt2.roots()[np.isreal(dpoly_dt2.roots())])
    # constrain extrema to the domain we are interested
    poly_extrema = poly_extrema[(time[0] < poly_extrema) & (poly_extrema < time[-1])]
    poly_turn_p = poly_turn_p[(time[0] < poly_turn_p) & (poly_turn_p < time[-1])]
    # get extrema with lowest y-value
    left_extrema_idx = np.argmin(poly(poly_extrema))
    if left_extrema_idx == poly_extrema.size - 1:
        return None
    # get turning point to the right of extrema
    turn_p_idx = np.argmax(poly_turn_p > poly_extrema[left_extrema_idx])
    slope_turn_p = dpoly_dt1(poly_turn_p[turn_p_idx])
    # get regime that is similar to the slope of turning point
    time_interval = np.linspace(
        poly_extrema[left_extrema_idx],
        poly_extrema[left_extrema_idx+1],
        num=time.size
    )
    rel_slope_difference = np.abs((dpoly_dt1(time_interval) - slope_turn_p) / slope_turn_p)
    time_bounds = time_interval[(rel_slope_difference < allowed_slope_deviation)][[0,-1]]
    time_bounds[0] = poly_turn_p[turn_p_idx]
    # compute fit on the interval that we found
    fit_interval_idx = [
        np.argmin(np.abs(time - time_bounds[0])),
        np.argmin(np.abs(time - time_bounds[1]))
    ]
    fit_interval_slice = slice(*fit_interval_idx)
    fit_result: LinregressResult = stats.linregress(
        time[fit_interval_slice],
        np.log(field_energy[fit_interval_slice]),
        alternative='less'
    )
    # abort if the slope is negative of the interval-length is weird
    if (
        fit_result.slope <= 0.0
        or slope_turn_p <= 0.0
        or time_bounds[1] - time_bounds[0] <= min_interval_length
    ):
        return None
    return fit_result, fit_interval_idx, (
        poly,
        poly_extrema[left_extrema_idx:left_extrema_idx+2],
        poly_turn_p[turn_p_idx]
    )

def avgTemperatureFromMomentumDist(
    momentum_grid: NDArray,
    dist_x_px: NDArray,
    info: RunInfo,
    species: Species
) -> NDArray:
    s_info = info[species]

    dx = 0.5 * info.lambda_D
    dV = (momentum_grid[1] - momentum_grid[0]) / (s_info.mass * constants.electron_mass)

    f_boltz = dist_x_px / (dV * dx)
    v = momentum_grid / (s_info.mass * constants.electron_mass)
    n0 = np.trapezoid(x=v, y=f_boltz, axis=-1)
    u = np.trapezoid(x=v, y=v * f_boltz, axis=-1) / n0
    P = s_info.mass * constants.electron_mass * np.trapezoid(
        x=v, y=(u[...,np.newaxis] - v) ** 2 * f_boltz, axis=-1
    )
    T_electron = P / (constants.electron_volt * n0)

    return T_electron

def readFromRun(
    folder: Path,
    dataset_names: list[str],
    processElement: Callable[[h5py.Dataset], ArrayLike]|None=None,
    time_interval: slice|int=slice(None),
    recursive: bool=False,
) -> tuple[NDArray, tuple[NDArray,...], tuple[Path]]:
    """Extracts specified datasets from a single simulation run.

    Args:
        folder (Path): Folder that contains the simulation data
        dataset_names (list[str]): Names of the datasets to extract
        processElement (Callable[[h5py.Dataset], ArrayLike]|None): Function to process
            individual datasets. Defaults to None.
        time_interval (slice|int): Select a range of time-indices of interest.
            Defaults to slice(None).
        recursive (bool): Whether to recursively read all sub-folders. Defaults to False.
    Returns:
        tuple[NDArray,tuple[NDArray,...],list[Path]]: Time first, then datasets in the same
            order as provided. Finally, the folder(s) that contain the simulation data are
            returned.
    """
    if recursive:
        return _readFromMultipleRuns(
            folder, dataset_names,
            processElement=processElement,
            time_interval=time_interval
        )

    files = sorted(folder.glob("*.h5"))
    assert len(files) > 0, f"ERROR: No files in directory '{folder}'"
    if len(dataset_names) == 0:
        print("WARNING: No quantities selected")

    if processElement is None:
        processElement = lambda x: np.squeeze(x)

    if isinstance(time_interval, int):
        file_idx = time_interval
        quantities = []
        with h5py.File(files[file_idx]) as h5_file:
            time = np.array(h5_file["Header"].attrs["time"])
            for i, key in enumerate(dataset_names):
                quantities.append(processElement(h5_file[key]))
        return time, (q for q in quantities), [folder]

    time = np.empty(len(files[time_interval]))
    quantities = [[] for _ in dataset_names]

    for file_idx, file_path in enumerate(files[time_interval]):
        with h5py.File(file_path) as h5_file:
            time[file_idx] = h5_file["Header"].attrs["time"][()]

            for i, key in enumerate(dataset_names):
                quantities[i].append(processElement(h5_file[key]))
    return time, (np.array(q) for q in quantities), [folder]

def _readFromMultipleRuns(
    folder: Path,
    dataset_names: list[str],
    processElement: Callable[[h5py.Dataset], ArrayLike]|None=None,
    time_interval: slice|int=slice(None)
) -> tuple[NDArray, tuple[NDArray,...], tuple[Path]]:
    """Read data from multiple runs (directories)

    Args:
        folder (Path): Current folder. Can contain other folders.
        dataset_names (list[str]): Names of the datasets to extract
        processElement (Callable[[h5py.Dataset], ArrayLike] | None, optional): Function to process
            individual datasets. Defaults to None.
        time_interval (slice | int, optional): Select a range of time-indices of interest.
            Defaults to slice(None).

    Returns:
        tuple[NDArray,tuple[NDArray,...],list[Path]]: Time first, then datasets in the same
            order as provided. Finally, the folder(s) that contain the simulation data are
            returned.
    """
    files = sorted(folder.glob("*.h5"))
    if len(files) > 0:
        return readFromRun(folder, dataset_names, processElement, time_interval)

    sub_folders = sorted(
        path for path in folder.iterdir()
        if path.is_dir() and len(list(path.glob("**/*.h5", recurse_symlinks=True))) > 0
    )
    assert len(sub_folders) > 0, "Found no simulation data"
    time_runs = []
    quantities_runs = []
    folders_runs = []
    for folder_path in sub_folders:
        time, quantities, folders = _readFromMultipleRuns(
            folder_path, dataset_names, processElement, time_interval
        )
        time_runs.append(time)
        quantities_runs.append(list(quantities))
        folders_runs.append(folders)
    # Make sure that the format is the same across runs
    assert all(time_runs[0].shape == t.shape for t in time_runs), "Times have to match across runs"
    assert all(np.all(time_runs[0] == t) for t in time_runs), "Times have to match across runs"
    assert all(
        all(ref_quantity.shape == quantities[q_idx].shape for quantities in quantities_runs)
        for q_idx, ref_quantity in enumerate(quantities_runs[0])
    ), "Shape of datasets has to match across runs"
    time = time_runs[0]
    quantities = (
        np.array([quantities[q_idx] for quantities in quantities_runs])
        for q_idx in range(len(dataset_names))
    )
    return time, quantities, folders_runs

def estimateFrequency(
    axis: int,
    axis_grid: NDArray,
    E_field: NDArray,
    peak_cutoff: float=0.95
) -> tuple[NDArray|float, float]:
    """Estimate frequency in some direction.

    Args:
        axis (int): Axis in which to perform FFT.
        axis_grid (NDArray): Grid corresponding to FFT-axis.
        E_field (NDArray): Electric field (at least 2D).
        peak_cutoff (float, optional): Values around the peak to consider. Defaults to 0.95.

    Returns:
        tuple[NDArray|float, float]: Frequency and corresponding error.
    """
    assert 0.0 <= peak_cutoff <= 1.0, "Peak cutoff must be non-negative and less than or equal to 1"
    assert -2 <= axis <= -1, "Expect axis layout with spatial and temporal dimension"
    assert E_field.ndim >= 2, "E-field needs at least space and time dimensions"
    dx = abs(axis_grid[1] - axis_grid[0])
    fft = np.abs(np.fft.rfft(E_field, axis=axis)) ** 2
    mean_fft = np.mean(fft, axis=-2 if axis == -1 else -1)
    N = mean_fft.shape[-1]
    # compute fractional index of peak center
    peak_index = np.apply_along_axis(
        lambda x: np.mean(np.nonzero(x > peak_cutoff * np.max(x, axis=-1))[0]),
        axis=-1,
        arr=mean_fft
    )
    k = 2 * np.pi * peak_index / (dx * N)
    k_err = 2 * np.pi / (np.sqrt(2) * dx * N)
    return k, k_err
