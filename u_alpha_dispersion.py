from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from threading import RLock

import h5py
import numpy as np
from scipy import optimize, constants, special
from tqdm import tqdm

from basic import RunInfo, SpeciesInfo

def propertiesAtMaxGrowthRate(
    Vab: float, cs: float,
    ve: float, vp: float, va: float,
    omega_pe: float, omega_pp: float, omega_pa: float,
    theta_arr: np.ndarray, k_arr: np.ndarray, kz_mat: np.ndarray,
):
    omega_arr   = np.empty_like(k_arr, dtype=np.complex128)
    omega_mat   = np.full_like(kz_mat, fill_value=np.nan, dtype=np.complex128)

    # calculate dispersion on grid
    for theta_idx, kz_arr in enumerate(tqdm(kz_mat, leave=False)):
        for k_idx, (k, kz) in enumerate(zip(k_arr, kz_arr)):
            xip = lambda omega: omega / (k * vp)
            xie = lambda omega: omega / (k * ve)
            xia = lambda omega: (omega - kz * Vab) / (k * va)

            def disprel(omega):
                omega = omega[0] + 1j * omega[1]
                result = (
                    1
                    + 2 * omega_pp ** 2 / (k ** 2 * vp ** 2) * (
                        1 + 1j * np.sqrt(np.pi) * xip(omega) * special.wofz(xip(omega))) +
                    + 2 * omega_pe ** 2 / (k ** 2 * ve ** 2) * (
                        1 + 1j * np.sqrt(np.pi) * xie(omega) * special.wofz(xie(omega))) +
                    + 2 * omega_pa ** 2 / (k ** 2 * va ** 2) * (
                        1 + 1j * np.sqrt(np.pi) * xia(omega) * special.wofz(xia(omega)))
                )
                return [np.real(result), np.imag(result)]

            if k_idx < 10:
                omega0 = cs * k / 2 + 500j
            else:
                omega0 = omega_arr[k_idx-1] + 0.01j * omega_pp

            sol = optimize.root(
                disprel, x0=[np.real(omega0), np.imag(omega0)],
                tol=1e-12, method="lm", options={
                    'xtol': 1e-11,
            })

            omega_arr[k_idx] = sol.x[0] + 1j * sol.x[1]
        omega_mat[theta_idx] = omega_arr

    # find maximum
    gammas = np.imag(omega_mat)
    theta_idx, k_idx = np.unravel_index(np.argmax(gammas), gammas.shape)
    gamma_max = gammas[theta_idx, k_idx]
    if gamma_max < 0:
        return np.nan, np.nan, np.nan, np.nan
    theta_max = theta_arr[theta_idx]
    k_max = k_arr[k_idx]
    omega_max = np.real(omega_mat)[theta_idx, k_idx]
    return gamma_max, theta_max, k_max, omega_max

if __name__ == "__main__":
    OUTPUT_FILENAME = Path("theory_u_alpha_dispersion_v2.h5")
    info = RunInfo(
        electron=SpeciesInfo(
            number_density=12.0e6,
            temperature=100.0,
            charge=-1,
            mass=1.0,
            bulk_velocity=0.0
        ),
        proton=SpeciesInfo(
            number_density=10.0e6,
            temperature=3.0,
            charge=+1,
            mass=1836.152674,
            bulk_velocity=0.0
        ),
        alpha=SpeciesInfo(
            number_density=1.0e6,
            temperature=12.0,
            charge=+2,
            mass=7294.29953,
            bulk_velocity=np.nan
        )
    )
    assert not OUTPUT_FILENAME.exists(), "Cannot overwrite output-file."
    ua_bulk_arr = np.linspace(50e3, 200e3, num=76)
    theta_arr = np.linspace(0, 90, num=451) * np.pi / 180
    k_arr: np.ndarray = np.linspace(0.5, 1.0, num=251) / info.lambda_D
    kz_mat = k_arr * np.cos(theta_arr)[:,np.newaxis]

    # compute dispersion relation
    partial_func = partial(propertiesAtMaxGrowthRate,
        cs=info.c_s,
        ve=info.electron.v_thermal,
        vp=info.proton.v_thermal,
        va=info.alpha.v_thermal,
        omega_pe=info.electron.omega,
        omega_pp=info.proton.omega,
        omega_pa=info.alpha.omega,
        theta_arr=theta_arr, k_arr=k_arr, kz_mat=kz_mat
    )
    tqdm.set_lock(RLock())
    with ThreadPoolExecutor(
        max_workers=8,
        initializer=tqdm.set_lock,
        initargs=(tqdm.get_lock(),)
    ) as pool:
        futures = pool.map(partial_func, ua_bulk_arr)
        results = list(tqdm(futures, total=len(ua_bulk_arr)))

    # save results
    results = np.array(results).T
    gamma_max = results[0]
    theta_max = results[1]
    k_max = results[2]
    omega_max = results[3]
    with h5py.File(OUTPUT_FILENAME, mode="x") as f:
        f["search_space/theta"] = theta_arr
        f["search_space/theta"].attrs["unit"] = "radians"
        f["search_space/k"] = k_arr
        f["u_alpha_bulk"] = ua_bulk_arr
        f["gamma_max"] = gamma_max
        f["theta_max"] = theta_max
        f["k_max"]     = k_max
        f["omega_max"] = omega_max
        f.attrs["description"] = "Properties of the instability at the maximum growth rate for different alpha-bulk speeds. All other parameters are at nominal conditions (cf. Graham 2025). All values are in SI-units."
