from scipy import optimize, constants, special
import numpy as np

ne = 12e6 # 1/m^3
Te = 100 # eV
Ta = 12 # eV
Tp = 3 # eV
Vab = 100e3 # m/s

qe = constants.elementary_charge
ma = 4.0 * constants.proton_mass
mp = constants.proton_mass
me = constants.electron_mass

ve = np.sqrt(2 * qe * Te / me)
va = np.sqrt(2 * qe * Ta / ma)
vp = np.sqrt(2 * qe * Tp / mp)
cs = ve / np.sqrt(2) * np.sqrt(me / mp) * (1 + 3 * Tp / Te)

wpe = np.sqrt(ne * qe ** 2 / (me * constants.epsilon_0))
ld =  ve / (wpe * np.sqrt(2))

k_arr = np.arange(0.01, 1.5+0.001, 0.001) / ld

nanp_ratio = 10 ** np.arange(-3, 1+0.05, 0.05)
np_arr    = ne / (2 * nanp_ratio + 1)
na_arr    = (ne - np_arr) / 2
wpp_arr   = np.sqrt(np_arr * qe ** 2 / (mp * constants.epsilon_0))
wpa_arr   = np.sqrt(na_arr * (2 * qe) ** 2 / (ma * constants.epsilon_0))

omega_arr   = np.zeros_like(k_arr, dtype=np.complex128)
omega_prev_arr = np.zeros_like(k_arr, dtype=np.complex128)

gamma_max   = np.full_like(nanp_ratio, fill_value=np.nan)
k_max       = np.full_like(nanp_ratio, fill_value=np.nan)
omega_max   = np.full_like(nanp_ratio, fill_value=np.nan)

for ratio_idx, (n_p, n_a, wpp, wpa) in enumerate(zip(
    np_arr, na_arr, wpp_arr, wpa_arr
)):
    for k_idx in range(len(k_arr)):
        xip = lambda omega: omega / (k_arr[k_idx] * vp)
        xie = lambda omega: omega / (k_arr[k_idx] * ve)
        xia = lambda omega: (omega - k_arr[k_idx] * Vab) / (k_arr[k_idx] * va)

        def disprel(omega):
            omega = omega[0] + 1j * omega[1]
            result = (
                1 + 2 * wpp ** 2 / (k_arr[k_idx] ** 2 * vp ** 2) *
                (1 + 1j * np.sqrt(np.pi) * xip(omega) * special.wofz(xip(omega))) +
                + 2 * wpe ** 2 / (k_arr[k_idx] ** 2 * ve ** 2) * (
                    1 + 1j * np.sqrt(np.pi) * xie(omega) * special.wofz(xie(omega))) +
                + 2 * wpa ** 2 / (k_arr[k_idx] ** 2 * va ** 2) * (
                    1 + 1j * np.sqrt(np.pi) * xia(omega) * special.wofz(xia(omega)))
            )
            return [np.real(result), np.imag(result)]

        if k_idx < 4:
            if ratio_idx == 0:
                omega0 = cs * k_arr[k_idx] / 5
            else:
                omega0 = omega_prev_arr[k_idx]
        else:
            omega0 = omega_arr[k_idx-1]

        sol = optimize.root(
            disprel, x0=[np.real(omega0), np.imag(omega0)],
            tol=1e-12, method="lm", options={
                'xtol': 1e-11,
        })

        omega_arr[k_idx] = sol.x[0] + 1j * sol.x[1]
    omega_prev_arr = omega_arr.copy()

    idxgam = np.argmax(np.imag(omega_arr))
    gamma = np.imag(omega_arr)[idxgam]
    if gamma >= 0:
        gamma_max[ratio_idx] = gamma
        k_max[ratio_idx] = k_arr[idxgam]
        omega_max[ratio_idx] = np.real(omega_arr[idxgam])

import h5py

with h5py.File("theory_density_ratio.h5", mode="x") as f:
    f["na_np_ratio"] = nanp_ratio
    f["k_max"] = k_max
    f["omega_max"] = omega_max
    f["gamma_max"] = gamma_max
