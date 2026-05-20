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
    assert d_ps.ndim == 1
    new_shape = tuple(d_ps.size if d_ps.size == s else 1 for s in dist_x_px.shape)
    f_v = dist_x_px / d_ps.reshape(
        new_shape # (d_ps.shape + ((1,) * (dist_x_px.ndim - d_ps.ndim)))
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
    dv_x = dv_x.reshape(
        *[sz if sz in dv_x.shape else 1 for sz in dist_x_px.shape]
    ) 
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
    dv_x = dv_x.reshape(*[sz if sz in dv_x.shape else 1 for sz in dist_px_py.shape]) 
    dv_y = dv_y.reshape(*[sz if sz in dv_y.shape else 1 for sz in dist_px_py.shape]) 
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

import numpy as np

def subpixelPeak3dSpectrum(P, omega, kx, ky, reg_size):
    """
    Subpixel 3D quadratic peak interpolation for a 3D power spectrum P(kx, ky, omega).

    Args:
        P: 3D Power spectrum with shape (Nt, Nx, Ny)
        omega, kx, ky: 1D Coordinate axes corresponding to P
        reg_size: Size of the region around argmax which is considered.

    Returns
    -------
    omega_peak, kx_peak, ky_peak, std_omega, std_kx, std_ky: floats
        Subpixel-interpolated location of the spectral peak
    """

    # 1. Find integer-grid maximum index
    idx = np.unravel_index(np.argmax(P), P.shape)
    i, j, k = idx

    # Guard: cannot interpolate on edges
    if (
        i == 0 or i == P.shape[0]-1
        or j == 0 or j == P.shape[1]-1
        or k == 0 or k == P.shape[2]-1
    ): return omega[i], kx[j], ky[k]

    # Coordinates around peak (central finite difference grid)
    # relative coordinates: -1, 0, +1
    idx_grid = np.arange(-reg_size, reg_size+1)
    X, Y, Z = np.meshgrid(idx_grid, idx_grid, idx_grid, indexing="ij")
    X = X.ravel()
    Y = Y.ravel()
    Z = Z.ravel()

    # Values around peak
    P_local = P[
        i-reg_size:i+reg_size+1,
        j-reg_size:j+reg_size+1,
        k-reg_size:k+reg_size+1,
    ].ravel()

    # Construct design matrix for quadratic fit:
    # f = ax x^2 + ay y^2 + az z^2 + bx x + by y + bz z + c
    A = np.column_stack([
        X*X, Y*Y, Z*Z,
        X, Y, Z,
        np.ones_like(X)
    ])

    # Solve least squares for quadratic coefficients
    coeffs, residuals, rank, s = np.linalg.lstsq(A, P_local, rcond=None)

    ax, ay, az, bx, by, bz, c = coeffs

    # 2. Compute subpixel peak = stationary point of gradient of paraboloid
    # ∂f/∂x = 2 a_x x + b_x = 0 → x_peak = -b_x/(2a_x)
    # (same for y, z)
    def safe_peak(b, a):
        if a == 0:
            return 0.0
        return -b / (2*a)

    dx = safe_peak(bx, ax)
    dy = safe_peak(by, ay)
    dz = safe_peak(bz, az)

    # Limit subpixel shifts to [-1,1] (fit region)
    dx = np.clip(dx, -reg_size, reg_size)
    dy = np.clip(dy, -reg_size, reg_size)
    dz = np.clip(dz, -reg_size, reg_size)

    # 3. Convert shift from index-space to physical coordinates
    dw  = omega[1] - omega[0]
    dkx = kx[1] - kx[0]
    dky = ky[1] - ky[0]

    w_peak  = abs(omega[i] + dx * dw)
    kx_peak = abs(kx[j] + dy * dkx)
    ky_peak = abs(ky[k] + dz * dky)

    # Error calculations
    # Estimate noise variance
    if len(P_local) > len(coeffs):
        sigma2 = residuals[0] / (len(P_local)-len(coeffs))
    else:
        sigma2 = np.var(P_local - A @ coeffs)

    # Covariance matrix of parameters
    # C = sigma^2 (A^T A)^{-1}
    ATA_inv = np.linalg.inv(A.T @ A)
    C = sigma2 * ATA_inv

    dxdax =  bx/(2*ax**2)
    dxdbx = -1/(2*ax)

    var_x = (dxdax**2)*C[0,0] + (dxdbx**2)*C[3,3] + 2*dxdax*dxdbx*C[0,3]
    var_y = ( (by/(2*ay**2))**2 *C[1,1] +
              (-1/(2*ay))**2*C[4,4] +
              2*(by/(2*ay**2))*(-1/(2*ay))*C[1,4] )
    var_z = ( (bz/(2*az**2))**2 *C[2,2] +
              (-1/(2*az))**2*C[5,5] +
              2*(bz/(2*az**2))*(-1/(2*az))*C[2,5] )

    sigma_omega  = np.sqrt(var_x + 1 / 4) * abs(dw)
    sigma_kx = np.sqrt(var_y + 1 / 4) * abs(dkx)
    sigma_ky = np.sqrt(var_z + 1 / 4) * abs(dky)
    return w_peak, kx_peak, ky_peak, sigma_omega, sigma_kx, sigma_ky


def extractWaveProperties(info: RunInfo, files: list[Path]) -> dict[str,np.ndarray]:
    """Estimate all wave protperites (incl. theta) with errors from the
    simulations of given files.

    Args:
        files: The HDF5-files that contain the simulation results.

    Returns:
        A dictionary with wave properties and errors by u_alpha.
    """
    wave = {k: [] for k in ['u_alpha', 'k', 'omega', 'theta', 'k_err', 'omega_err', 'theta_err', 'theta_rms', 'theta_rms_err']}
    for file_idx, filename in enumerate(files):
        flow_velocity = int(filename.stem[-3:])
        wave['u_alpha'].append(flow_velocity)
        with h5py.File(filename) as f:
            E_x = f['Electric Field/ex'][1:]
            E_y = f['Electric Field/ey'][1:]
            time = f["Header/time"][1:] * info.omega_pp
            x = f["Grid/grid/X"] / info.lambda_D_electron
            y = f["Grid/grid/Y"] / info.lambda_D_electron
            if x.ndim > 1:
                assert np.all(x == x[0]) and np.all(y == y[0])
                x = x[0]
                y = y[0]
        
        res = fitGrowthRate(
            time,
            np.mean(E_x ** 2 + E_y ** 2, axis=(1,2)),
            allowed_slope_deviation=0.5,
        )
        regime = slice(res[1][-1])
        
        E = E_x[regime] + 1j * E_y[regime]
        E_k_omega = np.fft.fftn(E, axes=(0,1,2))
        P = np.fft.fftshift(np.abs(E_k_omega)**2, axes=(0,1,2))
        
        Nt, Nx, Ny = E.shape
        dt, dx, dy = time[1]-time[0], x[1]-x[0], y[1]-y[0]
        omega = np.fft.fftshift(np.fft.fftfreq(Nt, d=dt)) * 2 * np.pi
        kx = np.fft.fftshift(np.fft.fftfreq(Nx, d=dx)) * 2 * np.pi
        ky = np.fft.fftshift(np.fft.fftfreq(Ny, d=dy)) * 2 * np.pi
    
        # theta from E_rms
        E_rms_x = np.sqrt(np.mean(E_x[regime] ** 2))
        E_rms_x_err = np.std(E_x[regime])  / np.sqrt(E_x.size)
        E_rms_y = np.sqrt(np.mean(E_y[regime] ** 2))
        E_rms_y_err = np.std(E_y[regime]) / np.sqrt(E_y.size)
        theta_max = np.arctan(abs(E_rms_y) / abs(E_rms_x))
        theta_err = np.sqrt(
            (E_rms_y / (E_rms_x ** 2 + E_rms_y ** 2)) ** 2 * E_rms_x_err ** 2 +
            (E_rms_x / (E_rms_x ** 2 + E_rms_y ** 2)) ** 2 * E_rms_y_err ** 2
        )
        wave['theta_rms'].append(theta_max)
        wave['theta_rms_err'].append(theta_err)
    
        # wave properties with sub-pixel resolution estimates
        (
            omega_max,
            kx_max,
            ky_max,
            sigma_omega,
            sigma_kx,
            sigma_ky,
        ) = subpixelPeak3dSpectrum(P, omega, kx, ky, reg_size=1)
        theta_max = np.arctan(abs(ky_max) / abs(kx_max))
        theta_err = np.sqrt(
            (ky_max / (kx_max ** 2 + ky_max ** 2)) ** 2 * sigma_kx ** 2 +
            (kx_max / (kx_max ** 2 + ky_max ** 2)) ** 2 * sigma_ky ** 2
        )
        k_max = np.linalg.norm([kx_max, ky_max])
        k_err = np.linalg.norm(k_max * np.array([sigma_kx, sigma_ky]) / np.linalg.norm(k_max))
        wave['omega'].append(omega_max)
        wave['omega_err'].append(sigma_omega)
        wave['k'].append(k_max)
        wave['k_err'].append(k_err)
        wave['theta'].append(theta_max)
        wave['theta_err'].append(theta_err)
    
    wave = {k: np.array(v) for k, v in wave.items()}
    return wave