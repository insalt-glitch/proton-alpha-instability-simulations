import numpy as np
from scipy import stats, constants
from scipy.stats._stats_py import LinregressResult
from numpy.typing import ArrayLike, NDArray
from numpy.polynomial import Polynomial
from collections.abc import Callable, Generator
from typing import Any
from pathlib import Path
import h5py

from basic import RunInfo, Species, SpeciesInfo

def fitGrowthRate(
    time: NDArray,
    field_energy: NDArray,
    polynomial_degree: int=7,
    min_interval_length: float=5.0,
    allowed_slope_deviation: float=0.2,
    reverse_search_direction: bool=False,
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
    if reverse_search_direction:
        turn_p_idx = np.nonzero(poly_turn_p < poly_extrema[left_extrema_idx+1])[0][-1]
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
        # try again with right extrema first
        right_extrema_idx = np.argmax(poly(poly_extrema))
        if right_extrema_idx == 0:
            return None
        turn_p_idx = np.nonzero(poly_turn_p < poly_extrema[right_extrema_idx])[0][-1]
        slope_turn_p = dpoly_dt1(poly_turn_p[turn_p_idx])

        time_interval = np.linspace(
            poly_extrema[right_extrema_idx-1],
            poly_extrema[right_extrema_idx],
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

def _numberDensity1D(v: NDArray, f_v: NDArray) -> float:
    n0 = np.trapezoid(x=v, y=f_v, axis=-1)
    return n0

def _flowVelocity1D(v: NDArray, f_v: NDArray) -> NDArray:
    n0 = _numberDensity1D(v, f_v)
    u = np.trapezoid(x=v, y=v * f_v, axis=-1) / n0
    return u

def _pressureTensor1D(v: NDArray, f_v: NDArray, mass: float) -> NDArray:
    u = _flowVelocity1D(v, f_v)
    P = mass * np.trapezoid(
        x=v, y=(u[...,np.newaxis] - v) ** 2 * f_v, axis=-1
    )
    return P

def _temperature1D(v: NDArray, f_v: NDArray, mass: float) -> NDArray:
    n0 = _numberDensity1D(v, f_v)
    P = _pressureTensor1D(v, f_v, mass)
    T_electron = P / (constants.electron_volt * n0)
    return T_electron

def normalizeDistributionXPx1D(
    x_grid: NDArray,
    px_grid: NDArray,
    dist_x_px: NDArray,
    info: SpeciesInfo,
) -> tuple[NDArray,NDArray]:
    v = px_grid / (info.si_mass)
    dx = np.abs(x_grid[...,1] - x_grid[...,0])
    dv_x = np.abs(v[...,1] - v[...,0])
    d_ps = dv_x * dx
    f_v = dist_x_px / d_ps.reshape(
        (d_ps.shape + ((1,) * (dist_x_px.ndim - d_ps.ndim)))
    )
    return v, f_v

def normalizeDistributionXPx2D(
    x_grid: NDArray,
    y_grid: NDArray,
    px_grid: NDArray,
    dist_x_px: NDArray,
    info: SpeciesInfo,
) -> tuple[NDArray,NDArray]:
    v = px_grid / (info.si_mass)
    dx = np.abs(x_grid[...,1] - x_grid[...,0])
    dv_x = np.abs(v[...,1] - v[...,0])
    length_y = np.abs(y_grid[...,0] - y_grid[...,-1])
    f_v = dist_x_px / (dv_x * dx * length_y)
    return v, f_v

def flowVelocity1D(
    x_grid: NDArray,
    px_grid: NDArray,
    dist_x_px: NDArray,
    info: SpeciesInfo,
) -> NDArray:
    v, f_v = normalizeDistributionXPx1D(x_grid, px_grid, dist_x_px, info)
    return _flowVelocity1D(v, f_v)

def temperature1D(
    x_grid: NDArray,
    px_grid: NDArray,
    dist_x_px: NDArray,
    info: SpeciesInfo,
) -> NDArray:
    v, f_v = normalizeDistributionXPx1D(x_grid, px_grid, dist_x_px, info)
    return _temperature1D(v, f_v, info.si_mass)

def normalizeDistributionPxPy(
    x_grid: NDArray,
    y_grid: NDArray,
    px_grid: NDArray,
    py_grid: NDArray,
    dist_px_py: NDArray,
    info: SpeciesInfo,
) -> tuple[NDArray,NDArray,NDArray]:
    v_x = px_grid / info.si_mass
    v_y = py_grid / info.si_mass
    dv_x = v_x[...,1] - v_x[...,0]
    dv_y = v_y[...,1] - v_y[...,0]
    length_x = np.abs(x_grid[...,-1] - x_grid[...,0])
    length_y = np.abs(y_grid[...,-1] - y_grid[...,0])
    f_v = dist_px_py / (dv_x * dv_y * length_x * length_y) # s^2/m^4
    return v_x, v_y, f_v

def _numberDensity2D(v_x, v_y, f_v):
    n0 = np.trapezoid(np.trapezoid(f_v, v_y), v_x)
    return n0

def _flowVelocity2D(v_x, v_y, f_v):
    n0 = _numberDensity2D(v_x, v_y, f_v)
    u_x = np.trapezoid(v_x * np.trapezoid(f_v, v_y), v_x) / n0
    u_y = np.trapezoid(np.trapezoid(v_y * f_v, v_y), v_x) / n0
    return u_x, u_y

def _pressureTensor2D(v_x: NDArray, v_y: NDArray, f_v: NDArray, mass: float) -> tuple[NDArray,NDArray]:
    u_x, u_y = _flowVelocity2D(v_x, v_y, f_v)
    P_xx = mass * np.trapezoid(
        (u_x[...,np.newaxis] - v_x) ** 2 * np.trapezoid(f_v, v_y), v_x
    )
    P_yy = mass * np.trapezoid(
        (u_y[...,np.newaxis] - v_y) ** 2 * np.trapezoid(f_v, v_x, axis=-2), v_y
    )
    return P_xx, P_yy

def _temperature2D(v_x: NDArray, v_y: NDArray, f_v: NDArray, mass: float) -> NDArray:
    n0 = _numberDensity2D(v_x, v_y, f_v)
    P_xx, P_yy = _pressureTensor2D(v_x, v_y, f_v, mass)
    temperature = (P_xx + P_yy) / (2 * constants.electron_volt * n0)
    return temperature

def flowVelocity2D(
    x_grid: NDArray,
    y_grid: NDArray,
    px_grid: NDArray,
    py_grid: NDArray,
    dist_px_py: NDArray,
    info: SpeciesInfo,
) -> tuple[NDArray,NDArray]:
    v_x, v_y, f_v = normalizeDistributionPxPy(
        x_grid, y_grid, px_grid, py_grid, dist_px_py, info,
    )
    return _flowVelocity2D(v_x, v_y, f_v)

def temperature2D(
    x_grid: NDArray,
    y_grid: NDArray,
    px_grid: NDArray,
    py_grid: NDArray,
    dist_px_py: NDArray,
    info: SpeciesInfo,
) -> NDArray:
    v_x, v_y, f_v = normalizeDistributionPxPy(
        x_grid, y_grid, px_grid, py_grid, dist_px_py, info,
    )
    return _temperature2D(v_x, v_y, f_v, info.si_mass)

def waveVector2D(
    x_grid: NDArray,
    y_grid: NDArray,
    E_field: NDArray,
    regime: slice = slice(None),
) -> tuple[NDArray,NDArray]:
    regime_E_field = E_field[regime]
    k_x, k_x_err = estimateFrequency(
        axis=-2,
        axis_grid=x_grid,
        E_field=regime_E_field,
        n_spatial_dims=2
    )
    k_y, k_y_err = estimateFrequency(
        axis=-1,
        axis_grid=y_grid,
        E_field=regime_E_field,
        n_spatial_dims=2
    )
    k = np.array([k_x, k_y])
    k_err = np.array([k_x_err, k_y_err])
    return k, k_err

def waveAngle2DFromWaveVector(
    k: NDArray|list,
    k_err: NDArray|list
) -> tuple[NDArray|float,NDArray|float]:
    assert len(k) == 2, "Expected 2D wave-vector"
    assert len(k_err) == 2, "Expected componentwise error of 2D wave-vector"
    k_x = k[0]
    k_x_err = k_err[0]
    k_y = k[1]
    k_y_err = k_err[1]
    # Compute wave-angle theta
    theta = np.arctan(k_y / k_x)
    # Compute wave-angle error (gaussian error propagation)
    theta_err = np.sqrt(
        (k_y / (k_x ** 2 + k_y ** 2)) ** 2 * k_x_err ** 2 +
        (k_x / (k_x ** 2 + k_y ** 2)) ** 2 * k_y_err ** 2
    )
    return theta, theta_err

def waveAngle2DFromElectricField(
    E_field_x: NDArray,
    E_field_y: NDArray,
    regime: slice = slice(None)
) -> tuple[NDArray|float,NDArray|float]:
    E_rms_x = np.sqrt(np.mean(E_field_x[regime] ** 2))
    E_rms_x_err = np.std(E_field_x[regime])  / np.sqrt(E_field_x.size)
    E_rms_y = np.sqrt(np.mean(E_field_y[regime] ** 2))
    E_rms_y_err = np.std(E_field_y[regime]) / np.sqrt(E_field_y.size)
    return waveAngle2DFromWaveVector(
        [E_rms_x, E_rms_y],
        [E_rms_x_err, E_rms_y_err]
    )

def readFromVariation(
    folder: Path,
    dataset_names: list[str],
    processElement: Callable[[h5py.Dataset], ArrayLike]=lambda x: x,
    time_interval: slice|int=slice(None),
    recursive: bool=False,
) -> tuple[NDArray, list[NDArray], list[Path]]:
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
        tuple[NDArray,list[NDArray,...],list[Path]]: Time first, then datasets in the same
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
        print("WARNING: No datasets selected")

    if isinstance(time_interval, int):
        time_interval = slice(time_interval, time_interval+1)

    quantities = [[] for _ in dataset_names]

    for file_idx, file_path in enumerate(files):
        with h5py.File(file_path) as h5_file:
            if file_idx == 0:
                time = h5_file["Header/time"][time_interval]
            else:
                assert np.all(time == h5_file["Header/time"][time_interval]), "Time has to be the same across all simulations but differs for '{files[0]}' and '{file_path}'"
            for i, key in enumerate(dataset_names):
                quantities[i].append(
                    processElement(np.squeeze(h5_file[key][time_interval]))
                )
    return time, [np.array(q) for q in quantities], [folder]

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
        return readFromVariation(folder, dataset_names, processElement, time_interval)

    sub_folders = sorted(
        path for path in folder.iterdir()
        if path.is_dir() and len(list(path.glob("**/*.h5", recurse_symlinks=True))) > 0
    )
    sub_folders = [f for f in sub_folders if "8192" not in f.as_posix()]
    assert len(sub_folders) > 0, "Found no simulation data"
    time_runs = []
    quantities_runs = []
    folders_runs = []
    for folder_path in sub_folders:
        time, quantities, folders = _readFromMultipleRuns(
            folder_path, dataset_names, processElement, time_interval
        )
        time_runs.append(time)
        quantities_runs.append(quantities)
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
    n_spatial_dims: int = 1,
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
    assert n_spatial_dims >= 0, "Number of spatial diemnsions must be positive"
    assert -(n_spatial_dims + 1) <= axis <= -1, "Expect axis layout with spatial and temporal dimension"
    assert E_field.ndim >= (n_spatial_dims + 1), "E-field needs at least space and time dimensions"
    dx = abs(axis_grid[1] - axis_grid[0])
    fft = np.abs(np.fft.rfft(E_field, axis=axis)) ** 2
    N = E_field.shape[axis]
    # NOTE: Alternative estimation via weighted_mean(argmax_k(fft))
    # k_arr = 2 * np.pi * np.argmax(fft, axis=axis) / (dx * N)
    # weights = np.max(fft, axis=axis)
    # weights /= np.sum(weights, axis=tuple(-(i+1) for i in range(n_spatial_dims)))
    # k = np.sum(k_arr * weights, axis=tuple(-(i+1) for i in range(n_spatial_dims)))

    # estimate index of peak center
    mean_fft = np.mean(fft, axis=tuple(-(i+1) for i in range(n_spatial_dims + 1) if -(i+1) != axis))
    k = 2 * np.pi * np.argmax(mean_fft, axis=-1) / (dx * N)
    k_sys_err = np.pi / (dx * N)
    return k, k_sys_err
