from pathlib import Path

import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.legend_handler import HandlerTuple
from scipy import constants

import analysis
from basic import RunInfo, physics, Species, SpeciesInfo
from basic.paths import (
    THEORY_U_ALPHA_FILE,
    THEORY_DENSITY_RATIO_FILE,
    V_FLOW_VARIATION_FOLDER,
    PARTICLE_VARIATION_FOLDER,
    FOLDER_2D, MPLSTYLE_FILE
)
from .plots_1D import runInfoForDenistyRatio
from .plots_2D import _loadPxPyDistribution, potentialFromElectricField

PIC_FILES_PREV = sorted(list(
    V_FLOW_VARIATION_FOLDER.glob("*.h5")),
    key=lambda x: int(x.stem[-3:]))
PIC_FILES = sorted(list(filter(
    lambda x: x.stem[-3:] not in ["150", "160"],
    V_FLOW_VARIATION_FOLDER.glob("*.h5")))
    + [FOLDER_2D / "special_grid_v_alpha_bulk_150.h5"],
    key=lambda x: int(x.stem[-3:]))

def flowVelocityWaveFrequency(ax, theory_u_norm, theory_omega, pic_data, theory_style, sim_style, cap_style):
    ax.plot(theory_u_norm, theory_omega, label=r"Linear theory", **theory_style)
    eb = ax.errorbar(pic_data['u_norm'], pic_data['omega'], yerr=pic_data['omega_err'], label=r"DFT (PIC)", **sim_style)
    _ = [cap.set(**cap_style) for cap in eb[1]]
    ax.set(
        ylabel = r"Wave frequency $\omega_\text{max}\,/\,\omega_\text{pp}$",
    )
    ax.legend()

def flowVelocityWaveVector(ax, theory_u_norm, theory_k, pic_data, theory_style, sim_style, cap_style):
    ax.plot(theory_u_norm, theory_k, label=r"Linear theory", **theory_style)
    eb = ax.errorbar(pic_data['u_norm'], pic_data['k'], yerr=pic_data['k_err'], label=r"DFT (PIC)", **sim_style)
    _ = [cap.set(**cap_style) for cap in eb[1]]
    ax.set(
        ylabel = r"Wave number $k_\text{max}\,\lambda_\text{D}$",
    )
    ax.legend()

def flowVelocityWaveAngle(ax, theory_u_norm, theory_theta, pic_data, theory_style, sim_style, cap_style):
    ax.plot(theory_u_norm, theory_theta, label=r"Linear theory", **theory_style)
    eb = ax.errorbar(
        pic_data['u_norm'],
        pic_data['theta'] * 180 / np.pi,
        yerr=pic_data['theta_err'] * 180 / np.pi,
        label=r"DFT (PIC)", **sim_style)
    _ = [cap.set(**cap_style) for cap in eb[1]]
    eb = ax.errorbar(
        pic_data['u_norm'],
        pic_data['theta_rms'] * 180 / np.pi,
        yerr=pic_data['theta_rms_err'] * 180 / np.pi,
        label=r"$E_\text{rms}$ (PIC)",
        **(sim_style | dict(marker='s', mfc='white', mec='darkorange', ms=5, zorder=5)))
    _ = [cap.set(**(cap_style | dict(mec='darkorange'))) for cap in eb[1]]
    ax.set(
        ylabel=r"Wave angle $\theta_\text{max}$ (deg)",
        yticks=np.arange(0, 70, 15),
    )
    ax.legend(loc='lower right')

def flowVelocityTrueVsPred(ax, u_crit_norm, pic_data, theory_style, sim_style, cap_style):
    ax.plot([0, 1e10], [0, 1e10], ls='--', lw=1, color='black', label=r"Identity")
    eb = ax.errorbar(
        pic_data['u_norm'][1:],
        u_crit_norm / np.cos(pic_data['theta'][1:]),
        yerr=u_crit_norm * np.abs(np.sin(pic_data['theta'][1:]) * pic_data['theta_err'][1:]),
        label=r"DFT (PIC)", **sim_style)
    _ = [cap.set(**cap_style) for cap in eb[1]]
    eb = ax.errorbar(
        pic_data['u_norm'][1:],
        u_crit_norm / np.cos(pic_data['theta_rms'][1:]),
        yerr=u_crit_norm * np.abs(np.sin(pic_data['theta_rms'][1:]) * pic_data['theta_rms_err'][1:]),
        label=r"$E_\text{rms}$ (PIC)",
         **(sim_style | dict(marker='s', mfc='white', mec='darkorange', ms=5, zorder=5)))
    _ = [cap.set(**(cap_style | dict(mec='darkorange'))) for cap in eb[1]]
    ax.set(
        ylim=(0.9, 2),
        ylabel=r"Flow velocity $u_\alpha^\text{(pred)}\,/\,u_\alpha^\text{(crit)}$",
    )
    ax.legend()

def flowVelocityWavePropsMosaic(info: RunInfo, pic_wave_props):
    with h5py.File(THEORY_U_ALPHA_FILE) as f:
        u_alpha = f["u_alpha_bulk"][:] * 1e-3
        gamma = f["gamma_max"][:] / info.omega_pp
        theta = f["theta_max"][:] * 180 / np.pi
        k_vec = f["k_max"][:] * info.lambda_D_electron
        omega = f["omega_max"][:] / info.omega_pp
    
    u_crit = 99.12905164474424
    vel_norm = u_crit
    u_norm = u_alpha / vel_norm
    pic_wave_props['u_norm'] = pic_wave_props['u_alpha'] / vel_norm
    u_crit_norm = u_crit / vel_norm
    
    # these are the means for omega and k; useful to write about for the text. Maybe we also want to add these to the plots
    omega_mean = np.mean(pic_wave_props['omega'][1:])
    omega_mean_err = np.sqrt(np.sum(pic_wave_props['omega_err'][1:] ** 2 / (pic_wave_props['omega_err'].size - 2)))
    k_mean = np.mean(pic_wave_props['k'][1:])
    k_mean_err = np.sqrt(np.sum(pic_wave_props['k_err'][1:] ** 2) / (pic_wave_props['k_err'].size - 2))
    # print(np.mean(omega[u_alpha>110]), omega_mean, omega_mean_err)
    # print(np.mean(k_vec[u_alpha>110]), k_mean, k_mean_err)
    # print(omega / k_vec * info.omega_pp * info.lambda_D)
    
    theory_style=dict(ls='--', color='black', lw=1)
    sim_style=dict(ls='', marker='o', mfc='white', mec='black', mew=1.2, elinewidth=2, capsize=3, ecolor="black")
    mean_style=dict(ls=':', color='olivedrab', lw=2)
    cap_style=dict(mew=2, mec="black")
    
    fig, axes = plt.subplots(2, 2, figsize=(6, 5.5), sharex=True, constrained_layout=True)
    ax = axes[0,0]
    flowVelocityWaveFrequency(ax, u_norm, omega, pic_wave_props, theory_style, sim_style, cap_style)
    ax.set(
        xlim = (0.9, 2),
        xticks=np.arange(1, 2.1, 0.2),
        ylim = (0.4, 0.95),
        yticks=np.arange(0.4, 1, 0.1),
    )
    ax = axes[0,1]
    flowVelocityWaveVector(ax, u_norm, k_vec, pic_wave_props, theory_style, sim_style, cap_style)
    ax.set(
        ylim=(0.6, 1.4),
        yticks=np.arange(0.6, 1.44, 0.2),
    )
    ax = axes[1,0]
    flowVelocityWaveAngle(ax, u_norm, theta, pic_wave_props, theory_style, sim_style, cap_style)
    ax.set(
        xlabel=r"Flow velocity $u_\alpha^{t=0}\,/\,u_\alpha^\text{(crit)}$",
    )
    ax = axes[1,1]
    eFieldVsFlowVelocity(ax, info, vel_norm, sim_style, cap_style)
    # flowVelocityTrueVsPred(ax, u_crit_norm, pic_wave_props, theory_style, sim_style, cap_style)
    ax.set(
        xlabel=r"Flow velocity $u_\alpha^{t=0}\,/\,u_\alpha^\text{(crit)}$",
    )
    for i, ax in enumerate(axes.flatten()):
        ax.set_aspect(np.ptp(ax.get_xlim()) / np.ptp(ax.get_ylim()))
        ax.text(0.05, 0.93,
            horizontalalignment='left',
            verticalalignment='top',
            s=rf"({chr(ord('a')+i)})",
            transform=ax.transAxes
        )
    for ax in axes[0]:
        ax.tick_params(top=False)
        ax_secondary = ax.secondary_xaxis('top', functions=(lambda x: x * vel_norm, lambda x: x / vel_norm))
        ax_secondary.set(
            xlabel=r'Flow velocity $u_\alpha^{t=0}$ (km$\,/\,$s)',
            xticks=np.arange(100, 202, 20),
        )

def electricField2DSnapshot(ax, flow_velocity: int, info: RunInfo, time: float|int, limit: float):
    filename = next(V_FLOW_VARIATION_FOLDER.glob(f"*{flow_velocity}.h5"))
    assert filename is not None
    with h5py.File(filename) as f:
        if isinstance(time, float):
            time_step = np.argmin(np.abs(f["Header/time"][:] * info.omega_pp - time))
        else:
            assert abs(time) < f["Header/time"].size, "Time out of range"
            time_step = time
        x = f["Grid/grid/X"] / info.lambda_D_electron
        y = f["Grid/grid/Y"] / info.lambda_D_electron

        if x.ndim > 1:
            assert np.all(x == x[0]) and np.all(y == y[0])
            x = x[0]
            y = y[0]
        E_x = f['Electric Field/ex'][time_step]
    E_x_max = np.max(np.abs(E_x))
    # print(np.percentile(np.abs(E_x), 99))
    mesh = ax.pcolormesh(x, y, E_x.T, cmap="bwr", rasterized=True, vmin=-limit, vmax=limit)
    ax.set(
        ylabel=("Position y$\\,/\\,\\lambda_\\text{D}$"),
        xticks=(np.arange(0, np.max(x)+1, 8)),
        yticks=(np.arange(0, np.max(y)+1, 8)),
        aspect=('equal'),
    )
    return mesh

def energyEFieldOverTime(ax, velocity: int, info: RunInfo, show_fit_details):
    filename = next(V_FLOW_VARIATION_FOLDER.glob(f"*{velocity}.h5"))
    with h5py.File(filename) as f:
        time = f["Header/time"][1:] * info.omega_pp
        energy = f["Electric Field/ex"][1:] ** 2
    energy = np.mean(
        energy,
        axis=tuple(range(1, energy.ndim))
    ) * (constants.epsilon_0 / 2) / (constants.electron_volt)
    fit_result = analysis.fitGrowthRate(time, energy)
    
    ax.plot(time, energy, label="$\\langle W_E\\rangle_\\mathbf{r}^\\text{sim}$ PIC",
             color="black", lw=2, zorder=2)
    assert fit_result is not None
    lin_fit, fit_interval, poly_info = fit_result
    ax.plot(
        time[slice(*fit_interval)], energy[slice(*fit_interval)],
        color="darkorange", lw=3, ls="solid", zorder=3,
        label="Linear reg.",
    )
    ax.plot(
        time, np.exp(lin_fit.slope * time + lin_fit.intercept),
        ls="--", color="cornflowerblue", zorder=9, lw=1.5,
        label=r"$\propto\exp(2\gamma\,t)$",
    )
    if show_fit_details:
        poly, extrema, turn_p = poly_info
        ax.plot(
            time, np.exp(poly(time)), label="Polynomial fit",
            color="gray", lw=1.5, ls=":", zorder=2, alpha=0.8,
        )
        ax.plot(
            extrema, np.exp(poly(extrema)), label="Turning points",
            color="gray", zorder=3, ls="", alpha=0.8,
            marker="o", markeredgecolor="black", markeredgewidth=1.5, markersize=10,
        )
        ax.plot(
            turn_p, np.exp(poly(turn_p)), label="Inflection point",
            color="white", zorder=3, ls="", marker="p",
            markersize=10, markeredgecolor="black", markeredgewidth=1.5,
        )
    ax.set(
        xlim=(0, 150),
        xticks=np.linspace(0.0, 150.0, num=6),
        yscale="log",
        xlabel="Time $t\\,\\omega_\\text{pp}$",
        ylabel="Energy $\\langle W_E\\rangle_x$ (eV$\\,/\\,$m$^3$)",
    )
    ax.legend()

def speciesTemperatureOverTime(ax, species, info):
    files = sorted(V_FLOW_VARIATION_FOLDER.glob("*.h5"))
    for filename in files:
        u_alpha = int(filename.stem[-3:])
        if u_alpha not in [95, 140]:
            continue
        with h5py.File(filename) as f:
            time = f["Header/time"][:]
            px_dist = np.mean(f[f"/dist_fn/x_px/{species.value}"], axis=1)
            py_dist = np.mean(f[f"/dist_fn/x_py/{species.value}"], axis=1)
            x_grid = f[f"Grid/x_px/{species.value}/X"][:]
            px_grid = f[f"Grid/x_px/{species.value}/Px"][:]
            py_grid = f[f"Grid/x_py/{species.value}/Py"][:]
            temp_3d = physics.kelvinToElectronVolt(np.mean(f[f'Derived/Temperature/{species.value}'][:], axis=(1,2)))
        
        time *= info.omega_pp
        t_x = analysis.temperature1D(x_grid, px_grid, px_dist, info[species])
        t_y = analysis.temperature1D(x_grid, py_grid, py_dist, info[species])
        temp_3d = (t_x + 2 * t_y) / 3 # Plot t_x + 2 * t_y instead
        temp_3d = np.convolve(temp_3d, np.ones(40)/40, mode='valid')
        t_x = np.convolve(t_x, np.ones(40)/40, mode='valid')
        t_y = np.convolve(t_y, np.ones(40)/40, mode='valid')
        time = time[20:-19]
    
        if u_alpha == 95:
            color='black'
        else:
            color='cornflowerblue'
        t_norm = 1# info[species].temperature
        ax.plot(time, t_x / t_norm, label=f"$T_{{x}}$", ls='--', color=color)
        ax.plot(time, t_y / t_norm, label=f"$T_{{y}}$", ls=':', color=color)
        ax.plot(time, temp_3d / t_norm, label=rf"$\tilde{{T}}$", ls='-', color=color)
    ax.set(
        xlim=(0, 150),
        ylabel=rf'Temperature $T_{{{species.symbol()}}}$ (eV)'
    )
    def conv(t):
        def f(x):
            return x / t
        return f
    def conv_inv(t):
        def f(x):
            return x * t
        return f
    ax_extra = ax.secondary_yaxis('right', functions=(conv(info[species].temperature), conv_inv(info[species].temperature)))
    ax_extra.set_ylabel(rf'Normalized $T_{{{species.symbol()}}}\,/\,T_{{{species.symbol()}}}^{{t=0}}$')
    return ax_extra

def temperatureElectricFieldMosaic(info):
    time = 55.0
    limit= 0.6
    fig = plt.figure(figsize=(6.8, 6.4), constrained_layout=True)
    outer_grid = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.3], wspace=0)
    # ------- left subplot -------- 
    left_gs = GridSpecFromSubplotSpec(
        3, 1,
        subplot_spec=outer_grid[0],
        height_ratios=[40 * 1.11, 56, 48],
    )
    ax_1 = fig.add_subplot(left_gs[0])
    ax_2 = fig.add_subplot(left_gs[1], sharex=ax_1)
    ax_3 = fig.add_subplot(left_gs[2])
    ax = np.array([ax_1, ax_2, ax_3])
    plt.setp(ax[0].get_xticklabels(), visible=False)
    # fig, ax = plt.subplots(2,1, sharex=True, height_ratios=[40 * 1.11, 56], constrained_layout=True)
    _ = electricField2DSnapshot(ax[0], 95, info, time, limit)
    mesh = electricField2DSnapshot(ax[1], 140, info, time, limit)
    ax[0].text(
        0.05, 0.95,
        horizontalalignment='left', verticalalignment='top',
        s=rf"a) $u_\alpha=95\,$km$\,/\,$s", # $t\\,\\omega_\\text{{pp}}=\\,${time:.0f}",
        transform=ax[0].transAxes,
        bbox=dict(facecolor='white', alpha=0.5)
    )
    ax[1].text(
        0.05, 0.95,
        horizontalalignment='left', verticalalignment='top',
        s=rf"b) $u_\alpha=140\,$km$\,/\,$s", #$t\\,\\omega_\\text{{pp}}=\\,${time:.0f}",
        transform=ax[1].transAxes,
        bbox=dict(facecolor='white', alpha=0.5)
    )
    
    divider = make_axes_locatable(ax[0])
    cax: plt.Axes = divider.append_axes("top", size="7%", pad=0.05)
    plt.colorbar(mesh, label="Electric field E$_x$ (V/m)", cax=cax, orientation='horizontal', location='top')
    cax.set_xticks(np.linspace(-limit, limit, num=5))
    ax[1].set_xlabel(r"Position x$\,/\,\lambda_\text{D}$")
    energyEFieldOverTime(ax[2], 140, info, False)
    ax[2].set(
        ylim=(1e4, 2e7),
        aspect=150/4.4,
    )
    ax[2].legend(loc='lower right', borderaxespad=0.1, frameon=False)
    ax[2].text(
        0.05, 0.95,
        horizontalalignment='left', verticalalignment='top',
        s=r"(c) $u_\alpha=140\,$km$\,/\,$s",
        transform=ax[2].transAxes,
        bbox=dict(facecolor='white', alpha=0.5),
        zorder=10
    )
    
    # ------- right subplot --------
    right_gs = GridSpecFromSubplotSpec(3, 1, subplot_spec=outer_grid[1])
    ax1 = fig.add_subplot(right_gs[0])
    ax2 = fig.add_subplot(right_gs[1], sharex=ax1)
    ax3 = fig.add_subplot(right_gs[2], sharex=ax1)
    axes = np.array([ax1, ax2, ax3])
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    for ax, species in zip([ax1, ax2, ax3], [Species.ELECTRON, Species.PROTON, Species.ALPHA]):
        ax_extra = speciesTemperatureOverTime(ax, species, info)
        h, l = ax.get_legend_handles_labels()
        match species:
            case Species.ELECTRON:
                text = '(d) Electrons'
                ax.set(ylim=(99.9, 101.2), yticks=np.arange(100, 101.3, 0.3))
                ax_extra.set_yticks(np.arange(1.00, 1.03, 0.003))
                custom_handles = [
                    Patch(facecolor='black', label=r'$u_\alpha^{t=0}=95\,$km/s'),
                    Patch(facecolor='cornflowerblue', label=r'$u_\alpha^{t=0}=140\,$km/s'),
                ]
                l1 = ax.legend(handles=custom_handles, loc=(0.25, 0.6), frameon=False,
                               labelspacing=0.1, edgecolor='black', borderpad=0.2)
                l2 = ax.legend(h[:3], l[:3], loc=(0.1, 0.26), frameon=False, 
                               labelspacing=0.1, edgecolor='black', borderpad=0.2)
                ax.add_artist(l1)
                ax.add_artist(l2)
            case Species.PROTON:
                text = '(e) Protons'
                ax_extra.set_yticks(np.arange(1, 7, 1))
                pass
            case Species.ALPHA:
                text = '(f) Alphas'
                ax_extra.set_yticks(np.arange(1, 6, 1))
        ax.tick_params(right=False)
        # ax.set(yticks=ax_extra.get_yticks() * info[species].temperature, yticklabels=[])
        ax.text(
            0.05, 0.95,
            horizontalalignment='left', verticalalignment='top',
            s=text,
            transform=ax.transAxes
        )
    ax3.set(xlabel="Time $t\\,\\omega_\\text{pp}$",)

def velocitySpaceGeometry(ax, u_alpha, label_prefix):
    v_ph = 69
    u_crit = 99
    v_res = u_crit - v_ph 
    if u_alpha is None:
        u_alpha = u_crit
    assert u_alpha >= u_crit, "negative not supported"
    theta = np.arccos(u_crit / u_alpha, out=np.array(0.0), where=v_ph / (u_alpha - v_res) < 1)
    # phase velocity circle and gradient circle
    alpha = np.linspace(0, 2 * np.pi, num=100)
    ax.plot(
        v_ph * np.sin(alpha), v_ph * np.cos(alpha), ls=":", lw=2,
        label=r"$\mathbf{v}^2=v_\text{ph}^2$", color="#000000"
    )
    ax.plot(
        u_alpha + v_res * np.sin(alpha), v_res * np.cos(alpha), ls=(0, (3,1,1,1)), lw=2,
        label=r"$(\mathbf{v}-\mathbf{u}_\alpha)^2=\xi^2 v_{\text{t}\alpha}^2$", color="cornflowerblue"
    )
    alpha = np.linspace(-theta, theta)
    ax.plot(0.65 * v_ph * np.cos(alpha), 0.65 * v_ph * np.sin(alpha), color="black", ls="-", lw=1.5)
    if u_alpha > u_crit:
        ax.text(
            v_ph / 8, 0,
            s=r"$2\theta_\text{max}$",
            horizontalalignment="left",
            verticalalignment="center",
            fontsize=9,
        )
    else:
        ax.text(
            v_ph / 8, v_ph / 6,
            s=r"$\theta_\text{max}=0$",
            horizontalalignment="left",
            verticalalignment="center",
            fontsize=9,
        )

    # proton and alpha max
    ax.plot(0,0, ls="", marker="o", color="white", mec='black', mew=1, label=r"$\mathbf{u}_\text{p}$ ($\mathbf{v}=0$)", zorder=10)
    ax.plot(u_alpha, 0, ls="", marker="p", markersize=9, color="cornflowerblue", label=r"$\mathbf{u}_\alpha$")
    # Interaction center
    s = np.array([-1_000, 1_000])
    ax.plot(
        v_ph * np.cos(theta) + np.sin(theta) * s,
        v_ph * np.sin(theta) - np.cos(theta) * s,
        ls="--", color="#900000"
    )
    if u_alpha > u_crit:
        ax.plot(
            +v_ph * np.cos(theta) + np.sin(theta) * s,
            -v_ph * np.sin(theta) + np.cos(theta) * s,
            ls="--", color="#900000"
        )
    # interaction regions
    width = 40
    rect_pos = plt.Rectangle(
        xy=(
            (v_ph - width / 2) * np.cos(theta) + np.sin(theta) * (-2 * v_ph),
            (v_ph - width / 2) * np.sin(theta) - np.cos(theta) * (-2 * v_ph)
        ),
        width=1000, height=width, angle=theta * 180 / np.pi+270, edgecolor="black", zorder=1, facecolor="#dc7800", alpha=0.7
    )
    if u_alpha > u_crit:
        rect_neg = plt.Rectangle(
            xy=(
                +(v_ph + width / 2) * np.cos(theta) + np.sin(theta) * (-2 * v_ph),
                -(v_ph + width / 2) * np.sin(theta) + np.cos(theta) * (-2 * v_ph)
            ),
            width=1000, height=width, angle=-theta * 180 / np.pi+90, edgecolor="black", zorder=1, facecolor="#dc7800", alpha=0.7
        )
    # arrows (v_ph)
    ann = ax.annotate(
        text='', xy=((v_ph + 3) * np.cos(theta), (v_ph + 3) * np.sin(theta)),
        xytext=(0,0), arrowprops=dict(arrowstyle='->', lw=2)
    )
    if u_alpha > u_crit:
        ann = ax.annotate(
            text='', xy=((v_ph + 3) * np.cos(theta), -(v_ph + 3) * np.sin(theta)),
            xytext=(0,0), arrowprops=dict(arrowstyle='->', lw=2)
        )
    # arrows (interaction)
    if u_alpha > u_crit:
        delta = 120
    else:
        delta = 60
    ann = ax.annotate(
        text='', xy=(
            (v_ph - width/2 - 3) * np.cos(theta) + np.sin(theta) * delta,
            (v_ph - width/2 - 3) * np.sin(theta) - np.cos(theta) * delta
        ),
        xytext=(
            (v_ph + width/2 + 3) * np.cos(theta) + np.sin(theta) * delta,
            (v_ph + width/2 + 3) * np.sin(theta) - np.cos(theta) * delta
        ), arrowprops=dict(arrowstyle='<->', lw=2, color="#693a00")
    )
    ann = ax.annotate(
        text='', xy=(
            (v_ph - width/2 - 3) * np.cos(theta) + np.sin(theta) * delta,
            -(v_ph - width/2 - 3) * np.sin(theta) + np.cos(theta) * delta
        ),
        xytext=(
            (v_ph + width/2 + 3) * np.cos(theta) + np.sin(theta) * delta,
            -(v_ph + width/2 + 3) * np.sin(theta) + np.cos(theta) * delta
        ), arrowprops=dict(arrowstyle='<->', lw=2, color="#693a00")
    )
    ax.text(
        0.02, 0.97,
        s=rf"{label_prefix} $u_\alpha {'=' if u_alpha == u_crit else '>'} v_\text{{ph}} + \xi v_{{\text{{t}}\alpha}}$",
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=0.5, pad=.2),
    )
    ax.add_patch(rect_pos)
    if u_alpha > u_crit:
        ax.add_patch(rect_neg)
    ax.set(
        aspect="equal",
        xlim=(-80, 180),
        ylim=(-100, 100),
        xticks=[-v_ph, 0, v_ph, 2*v_ph],
        xticklabels=["-1", "0", "1", "2"],
        yticks=[-v_ph, 0, v_ph],
        yticklabels=["-1", "0", "1"],
    )

def geometryParticleTransport(info):
    fig = plt.figure(figsize=(6, 5.6), constrained_layout=True)
    grid = GridSpec(2, 2, figure=fig, height_ratios=[0.7,1])
    ax1 = fig.add_subplot(grid[1, 0])
    axes = np.array([ax1, fig.add_subplot(grid[1, 1], sharey=ax1)])
    velocitySpaceGeometry(axes[0], 99, '(b)')
    velocitySpaceGeometry(axes[1], 140, '(c)')
    axes[0].set(ylabel=r"Velocity $v_y\,/\,v_\text{ph}$")
    for ax in axes:
        ax.set(
            xlabel=r"Velocity $v_x\,/\,v_\text{ph}$",
        )
    plt.setp(axes[1].get_yticklabels(), visible=False)

    ax = fig.add_subplot(grid[0, :])
    u_crit = 99.12905164474424
    theoryWaveProps(ax, info, u_crit)
    ax.set(
        xlim=(0.56, 2),
        ylim=(-0.05, 1.5),
        yticks=np.arange(0, 1.55, 0.3),
        aspect=0.9,
        xlabel=r"Flow velocity $u_\alpha\,/\,u_\alpha^\text{(crit)}$", 
    )
    [h, l] = ax.get_legend_handles_labels()
    legend1 = ax.legend(h[:2], l[:2], loc="upper right", labelspacing=0, borderaxespad=0.6)
    ax.legend(h[2:], l[2:], loc=(0.43, 0.15), labelspacing=0.1)
    ax.add_artist(legend1)
    ax.tick_params(top=False)
    ax_secondary = ax.secondary_xaxis('top', functions=(lambda x: x * u_crit, lambda x: x / u_crit))
    ax_secondary.set(
        xticks=np.arange(60, 210, 30),
        xlabel=r'Flow velocity $u_\alpha^{t=0}$ (km$\,/\,$s)',
    )
    ax.text(0.05, 0.95,
        horizontalalignment='left',
        verticalalignment='top',
        s="(a)",
        transform=ax.transAxes
    )
    
    [h, l] = axes[0].get_legend_handles_labels()
    fig.legend(
        [(h + [
            axes[0].scatter(1e3, 1e3, c='black', marker='$\\longrightarrow$', s=300),
            axes[0].scatter(1e3, 1e3, c='#693a00', marker='$⟷$', s=300)
        ])[i] for i in (4,5,3,2,0,1)],
        [(l+["$\\mathbf{v}_\\text{ph}$", r"$\partial_t f_s$"])[i] for i in (4,5,3,2,0,1)],
        ncols=1, labelspacing=0.1, loc=(0.425,0.11),
        columnspacing=0.5, borderaxespad=0.8, framealpha=0.8, fontsize=9)
    

def heatingVsFlowVelocity(ax, species: Species, info: RunInfo, vel_norm, normalize_temperature):
    n_points: int = 10
    files = PIC_FILES_PREV
    velocity = np.empty(len(files))
    T_diff = np.empty(len(files))
    T_diff_err = np.empty(len(files))
    for file_idx, filename in enumerate(files):
        velocity[file_idx] = int(filename.stem[-3:])
        with h5py.File(filename) as f:
            temp = physics.kelvinToElectronVolt(
                np.mean(f[f"Derived/Temperature/{species.value}"], axis=(1,2))
            )
            
        T_diff[file_idx] = np.mean(temp[-n_points:]) - np.mean(temp[:n_points])
        T_diff_err[file_idx] = np.sqrt(
            np.var(temp[-n_points:]) + np.var(temp[:n_points])
        ) / np.sqrt(n_points)
    if normalize_temperature:
        K_alpha_t0 = info.alpha.si_mass * info.alpha.number_density * (velocity * 1e3) ** 2 / (3 * constants.electron_volt * info[species].number_density)
        T_norm = K_alpha_t0 * 1e-2
    else:
        T_norm = info[species].temperature
    T_diff /= T_norm
    T_diff_err /= T_norm

    ax.errorbar(
        velocity / vel_norm, T_diff, yerr=T_diff_err,
        marker="o" if species == Species.PROTON else "s",
        markersize=8 if species == Species.PROTON else 7.5,
        color="cornflowerblue" if normalize_temperature else "white",
        ls="", markeredgecolor="black", markeredgewidth=1
    )
    if normalize_temperature:
        ax.set_ylabel(f"$\\Delta U_{species.symbol()}\\,/\\,K_\\alpha^{{t=0}}$ (%)")
    else:
        ax.set_ylabel(rf"$\Delta T_{species.symbol()}\,/\,T_{species.symbol()}^{{t=0}}$")

def eFieldVsFlowVelocity(ax, info, vel_norm, sim_style, cap_style):
    files = PIC_FILES_PREV
    velocity = np.empty(len(files))
    E_max = np.empty(len(files))
    E_max_err = np.empty(len(files))
    for file_idx, filename in enumerate(files):
        velocity[file_idx] = int(filename.stem[-3:])
        with h5py.File(filename) as f:
            E_x = f['Electric Field/ex'][:]
            E_y = f['Electric Field/ey'][:]
        # print(int(filename.stem[-3:]), np.max(np.sqrt(E_x ** 2 + E_y ** 2)))
        E = np.mean(np.sqrt(E_x ** 2 + E_y ** 2), axis=(1,2))
        max_idx = np.argmax(E)
        max_range = E[max_idx-5:max_idx+5]
        E_max[file_idx] = np.mean(max_range)
        E_max_err[file_idx] = np.std(max_range)
    velocity = np.array(velocity)
    E_max = np.array(E_max)
    E_max_err = np.array(E_max_err)

    eb = ax.errorbar(
        velocity / vel_norm, E_max, yerr=E_max_err, color="white",
        label='Electric field (PIC)', **sim_style,
    )
    _ = [cap.set(**cap_style) for cap in eb[1]]
    ax.set(
        ylabel=(r"$\max[\langle E\rangle_\mathbf{r}]_t$  (V/m)"),
        # yticks=np.arange(0.3, 0.46, 0.05),
    )
    # print(ax.get_ylim())
    ax.legend(fancybox=False, edgecolor="black", markerscale=0.7)

def energyEFieldVsFlowVelocity(ax, info, vel_norm):
    files = PIC_FILES_PREV
    velocity = np.empty(len(files))
    W_E_max = np.empty(len(files))
    W_E_max_err = np.empty(len(files))
    for file_idx, filename in enumerate(files):
        velocity[file_idx] = int(filename.stem[-3:])
        with h5py.File(filename) as f:
            E_x = f['Electric Field/ex'][:]
            E_y = f['Electric Field/ey'][:]
        W_E = np.mean(E_x ** 2 + E_y ** 2, axis=(1,2)) * (constants.epsilon_0 / 2) / constants.electron_volt
        max_idx = np.argmax(W_E)
        max_range = W_E[max_idx-5:max_idx+5]
        W_E_max[file_idx] = np.mean(max_range)
        W_E_max_err[file_idx] = np.std(max_range)
    velocity = np.array(velocity)
    W_E_max = np.array(W_E_max)
    W_E_max_err = np.array(W_E_max_err)
    # if normalize_energy:
    K_alpha_t0 = (info.alpha.si_mass * info.alpha.number_density * velocity ** 2 / (2 * constants.electron_volt))
    W_E_max /= K_alpha_t0
    W_E_max_err /= K_alpha_t0

    ax.errorbar(
        velocity / vel_norm, W_E_max * 1e-6, yerr=1e-6 * W_E_max_err, color="cornflowerblue",
        marker="d", markeredgecolor="black", markeredgewidth=1, markersize=8, ls="",
        label='E-field'
    )
    ax.set(
        ylabel=("$\\max[\\langle W_E\\rangle_\\mathbf{r}]_t\\,/\\,K_\\alpha^{t=0}$"),
        ylim=ax.get_ylim(),
    )
    y_min, y_max = ax.get_ylim()
    ax.legend(fancybox=False, edgecolor="black")

def theoryWaveProps(ax, info, vel_norm):
    with h5py.File(THEORY_U_ALPHA_FILE) as f:
        u_alpha = f["u_alpha_bulk"][:] * 1e-3
        gamma = f["gamma_max"][:] / info.omega_pp
        theta = f["theta_max"][:]
        k_vec = f["k_max"][:] * info.lambda_D_electron
        omega = f["omega_max"][:] / info.omega_pp

    ax.plot(
        u_alpha / vel_norm, gamma, color="cornflowerblue",
        label="$\\gamma_\\text{max}\\,/\\,\\omega_\\text{pp}$")
    ax.plot(
        u_alpha / vel_norm, omega, color="#888888",
        label="$\\omega_\\text{max}\\,/\\,\\omega_\\text{pp}$")
    ax.plot(
        u_alpha / vel_norm, k_vec, color="black",
        label="$k_\\text{max}\\,\\lambda_\\text{D}$")
    ax.plot(
        u_alpha / vel_norm, theta,
        label="$\\theta_\\text{max}$",
        color="orange", ls="-.", lw=2,
    )
    ax.set(
        ylabel="Wave properties",
    )

def flowVelocityEnergyTransferMosaic(info):
    vel_norm = 99.12905164474424 # u_crit
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 5), sharex=True, constrained_layout=True)
    
    for ax_row, normalize_temperature in zip(axes[0:], [False, True]):
        for ax, species in zip(ax_row, [Species.PROTON, Species.ALPHA]):
            heatingVsFlowVelocity(ax, species, info, vel_norm, normalize_temperature)
    for ax, species_name in zip(axes[0], ['Protons', 'Alphas']):
        ax.tick_params(top=False)
        ax_secondary = ax.secondary_xaxis('top', functions=(lambda x: x * vel_norm, lambda x: x / vel_norm))
        ax_secondary.set(
            xticks=np.arange(100, 190, 20),
            xlabel=f"$\\it{{{species_name}}}$\n" + r'Flow velocity $u_\alpha^{t=0}$ (km$\,/\,$s)',
        )
    for ax in axes[:,1]:
        ax.yaxis.set_label_position("right")   # move ylabel
        ax.yaxis.tick_right()                  # move tick labels to the right
        ax.tick_params(axis="y", which="both", left=True, right=True)
    for i, ax in enumerate(axes.flatten()):
        ax.text(0.05, 0.93,
            horizontalalignment='left',
            verticalalignment='top',
            s=rf"({chr(ord('a')+i)})",
            transform=ax.transAxes
        )
    axes[-1,0].set(
        xlim = (0.9, 1.9),
        xticks=np.arange(1, 2.0, 0.2),
        # ylim=(1.4,None),
    )
    for ax in axes[-1,:]:
        ax.set_xlabel(r"Flow velocity $u_\alpha^{t=0}\,/\,u_\alpha^\text{(crit)}$")
    axes[0,0].set_ylim(1.36, None)
    axes[0,0].set_ylabel("$\\it{Temperature}$\n" + axes[0,0].get_ylabel())
    axes[1,0].set_ylabel("$\\it{Kinetic}$ $\\it{energy}$\n" + axes[1,0].get_ylabel())
    fig.text(0, 0.5, r"$\it{Normalization}$", va='center', ha='right', rotation=90)

def convergenceFrequency(ax, info):
    time, (E_fields,), folders = analysis.readFromVariation(
        folder=PARTICLE_VARIATION_FOLDER,
        dataset_names=["/Electric Field/ex"],
        recursive=True
    )
    with h5py.File(next(PARTICLE_VARIATION_FOLDER.glob("*8192/rep_0.h5"))) as f:
        p8192_time = f["Header/time"][:]
        p8192_E_field = f["Electric Field/ex"][:]

    # fix units of time and energy
    time *= info.omega_pp
    p8192_time *= info.omega_pp
    # extract particle numbers
    particle_numbers = np.array([int(pfs[0].stem[-4:]) for pfs in folders])
    omega_arr = np.full(shape=E_fields.shape[:2], fill_value=np.nan)
    omega_err_arr = np.full_like(omega_arr, fill_value=np.nan)

    for p_idx, e_fields in enumerate(E_fields):
        if p_idx < 2:
            continue
        for r_idx, field in enumerate(e_fields):
            _, regime, _ = analysis.fitGrowthRate(
                time, np.mean(field ** 2, axis=1), allowed_slope_deviation=0.5
            )
            omega_arr[p_idx, r_idx], omega_err_arr[p_idx, r_idx] = analysis.estimateFrequency(
                -2, time, field[:(regime[-1])]
            )
    _, regime, _ = analysis.fitGrowthRate(
        p8192_time, np.mean(p8192_E_field ** 2, axis=1)
    )
    p8192_omega, p8192_omega_err = analysis.estimateFrequency(
        -2, p8192_time, p8192_E_field[:regime[-1]]
    )
    mean_omega = np.mean(omega_arr, axis=1)
    mean_omega_err = np.sqrt(np.sum((omega_err_arr / 4) ** 2, axis=1) + np.var(omega_arr, axis=1) / 4)
    p8192_omega_err = np.sqrt(p8192_omega_err ** 2 + np.mean(np.var(omega_arr, axis=1)[-2:]))
    l_vary = ax.errorbar(particle_numbers[2:], mean_omega[2:], yerr=mean_omega_err[2:],
        ls="", marker="o", color='white', lw=1.5,
        mec="black", mew=1.2, elinewidth=2, capsize=3, ecolor="black")
    _ = [cap.set(mew=2, mec="black") for cap in l_vary[1]]
    l_8192 = ax.errorbar(2 ** 13, p8192_omega, yerr=p8192_omega_err,
        ls="", marker="p", color='white', lw=1.5,
        mec="black", mew=1.2, elinewidth=2, capsize=3, ecolor="black", markersize=9)
    _ = [cap.set(mew=2, mec="black") for cap in l_8192[1]]
    with h5py.File(THEORY_U_ALPHA_FILE) as f:
        u_alpha = f["u_alpha_bulk"][:] * 1e-3
        omega = f["omega_max"][:] / info.omega_pp
    l_theory = ax.axhline(omega[np.argmin(np.abs(u_alpha - 100))], color="black", ls="--", lw=1)
    y_min, y_max = ax.set_ylim(0.7, 0.95)
    r_fail = ax.fill_between(
        [0, 2 ** 6.5],
        y1=y_min, y2=y_max,
        color="red", alpha=0.7
    )
    ax.set(
        yticks=np.linspace(0.7, 0.9, num=3),
        ylabel=r"Frequency $\omega_\text{max}\,/\,\omega_\text{pp}$",
    )
    ax.legend(
        [l_theory, (l_vary, l_8192), r_fail],
        ["Theory", "PIC (1D)", "No wave"], borderaxespad=0.3,
        loc="upper center", markerscale=0.7, framealpha=0.6,
        handler_map={tuple: HandlerTuple(ndivide=None)})
    return particle_numbers

def convergenceWavenumber(ax, info):
    time, (E_fields,), folders = analysis.readFromVariation(
        folder=PARTICLE_VARIATION_FOLDER,
        dataset_names=["/Electric Field/ex"],
        recursive=True
    )
    with h5py.File(next(PARTICLE_VARIATION_FOLDER.glob("*0032/rep_0.h5"))) as f:
        grid = np.squeeze(f["/Grid/grid"][0])

    with h5py.File(next(PARTICLE_VARIATION_FOLDER.glob("*8192/rep_0.h5"))) as f:
        p8192_time = f["Header/time"][:]
        p8192_E_field = f["Electric Field/ex"][:]
        p8192_grid = np.squeeze(f["Grid/grid"][0])
    # fix units of time and energy
    time *= info.omega_pp
    grid /= info.lambda_D_electron
    p8192_time *= info.omega_pp
    p8192_grid /= info.lambda_D_electron
    # extract particle numbers
    particle_numbers = np.array([int(pfs[0].stem[-4:]) for pfs in folders])
    k_arr = np.full(shape=E_fields.shape[:2], fill_value=np.nan)
    k_err_arr = np.full_like(k_arr, fill_value=np.nan)

    for p_idx, e_fields in enumerate(E_fields):
        if p_idx < 2:
            continue
        for r_idx, field in enumerate(e_fields):
            _, regime, _ = analysis.fitGrowthRate(
                time, np.mean(field ** 2, axis=1), allowed_slope_deviation=0.5)
            k_arr[p_idx, r_idx], k_err_arr[p_idx, r_idx] = analysis.estimateFrequency(
                -1, grid, field[:(regime[-1])]
            )
    _, regime, _ = analysis.fitGrowthRate(
        p8192_time, np.mean(p8192_E_field ** 2, axis=1)
    )
    p8192_k, p8192_k_err = analysis.estimateFrequency(
        -1, p8192_grid, p8192_E_field[:regime[-1]]
    )
    mean_k = np.mean(k_arr, axis=-1)
    mean_k_err = np.sqrt(np.sum((k_err_arr / 4) ** 2, axis=-1) + np.var(k_arr, axis=-1) / 4)
    p8192_k_err = np.sqrt(p8192_k_err ** 2 + np.mean(np.var(k_arr, axis=1)[-2:]))
    l_vary = ax.errorbar(particle_numbers[2:], mean_k[2:], yerr=mean_k_err[2:],
        ls="", lw=1.5, marker="o", color="white",
         mec="black", mew=1.2, elinewidth=2, capsize=3, ecolor="black")
    _ = [cap.set(mew=2, mec="black") for cap in l_vary[1]]
    l_8192 = ax.errorbar(8192, p8192_k, yerr=p8192_k_err,
        ls="", lw=1.5, color="white", marker="p",
        mec="black", mew=1.2, elinewidth=2, capsize=3, ecolor="black")
    _ = [cap.set(mew=2, mec="black") for cap in l_8192[1]]
    l_theory = ax.axhline(0.6604529941471382 * 1.472, color="black", ls="--", lw=1)
    y_min, y_max = ax.set_ylim(0.2, 1.15)
    r_fail = ax.fill_between(
        [0, 2 ** 6.5], y1=y_min, y2=y_max,
        color="red", alpha=0.7
    )
    ax.set(
        yticks=np.linspace(0.4, 1.0, num=3),
        ylabel="Wave number $k_\\text{max}\\,\\lambda_\\text{D}$",
    )
    ax.legend(
        [l_theory, (l_vary, l_8192), r_fail],
        ["Theory", "PIC (1D)", "No wave"], borderaxespad=0.3,
        loc=(0.44, 0.05), markerscale=0.7, labelspacing=0.4,
        handler_map={tuple: HandlerTuple(ndivide=None)})

def convergenceGrowthRate(ax, info):
    time, (energies,), folders = analysis.readFromVariation(
        folder=PARTICLE_VARIATION_FOLDER,
        dataset_names=["/Electric Field/ex"],
        processElement=lambda x: np.mean(np.array(x) ** 2, axis=1),
        time_interval=slice(0,751),
        recursive=True
    )
    with h5py.File(next(PARTICLE_VARIATION_FOLDER.glob("*8192/rep_0.h5"))) as f:
        p8192_time = f["Header/time"][:]
        p8192_energy = np.mean(f["Electric Field/ex"][:] ** 2, axis=1)

    time *= info.omega_pp
    energies *= constants.epsilon_0 / (2.0 * constants.electron_volt)
    p8192_time *= info.omega_pp
    p8192_energy *= constants.epsilon_0 / (2.0 * constants.electron_volt)
    # extract particle numbers
    particle_numbers = np.array([int(pfs[0].stem[-4:]) for pfs in folders])
    # extract growth rates from fits
    fits = [[analysis.fitGrowthRate(time, W_E) for W_E in es] for es in energies]
    p8192_growth_rate = analysis.fitGrowthRate(p8192_time, p8192_energy)[0].slope / 2
    p8192_growth_rate_err = np.abs(
        analysis.fitGrowthRate(p8192_time, p8192_energy, allowed_slope_deviation=0.4)[0].slope -
        analysis.fitGrowthRate(p8192_time, p8192_energy, allowed_slope_deviation=0.1)[0].slope
    ) / 2
    growth_rates = np.full(energies.shape[:2], np.nan)
    growth_rate_errs = np.full_like(growth_rates, np.nan)
    for p_idx, es in enumerate(energies):
        for r_idx, W_E in enumerate(es):
            res = analysis.fitGrowthRate(time, W_E)
            res_small = analysis.fitGrowthRate(time, W_E, allowed_slope_deviation=0.1)
            res_big = analysis.fitGrowthRate(time, W_E, allowed_slope_deviation=0.4)
            if None not in [res, res_small, res_big]:
                growth_rates[p_idx,r_idx] = res[0].slope / 2
                growth_rate_errs[p_idx,r_idx] = np.abs(res_small[0].slope - res_big[0].slope) / 2

    growth_rates_mean = np.mean(growth_rates, axis=1)
    growth_rates_err  = np.sqrt(np.sum((growth_rate_errs / 4) ** 2, axis=1) + np.var(growth_rates, axis=1) / 4)
    p8192_growth_rate_err = np.sqrt(p8192_growth_rate_err ** 2 + np.mean(np.var(growth_rates, axis=1)[-2:]))
    l_vary = ax.errorbar(
        particle_numbers, growth_rates_mean, yerr=growth_rates_err,
        ls="", lw=1.5, color="white", zorder=6, marker="o", 
        markersize=8, mec="black", mew=1.2, elinewidth=2, capsize=3, ecolor="black"
    )
    _ = [cap.set(mew=2, mec="black") for cap in l_vary[1]]
    l_8192 = ax.errorbar(
        8192, p8192_growth_rate, yerr=p8192_growth_rate_err,
        ls="", lw=1.5, color="white", marker="p", 
        markersize=9, mec="black", mew=1.2, elinewidth=2, capsize=3, ecolor="black"
    )
    _ = [cap.set(mew=2, mec="black") for cap in l_8192[1]]
    with h5py.File(THEORY_U_ALPHA_FILE) as f:
        gamma_max = f["gamma_max"][-1] / info.omega_pp
    l_theory = ax.axhline(gamma_max, color="black", ls="--", lw=1)
    y_min, y_max = ax.set_ylim(1e-2, 14e-2)
    r_fail = ax.fill_between(
        [0, 2 ** 6.5], y1=y_min, y2=y_max,
        color="red", alpha=0.7
    )
    ax.set(
        yticks=np.linspace(0.3e-1, 1.2e-1, 4),
        ylabel=r"Growth rate $\gamma\,/\,\omega_\text{pp}$",
    )
    from matplotlib.legend_handler import HandlerErrorbar
    legend = ax.legend([l_theory, (l_vary, l_8192), r_fail],
               [r"Theory $\gamma_\text{max}$", "PIC (1D)", "No wave"],
               loc=(0.07,0.43), markerscale=0.7,
               handler_map={tuple: HandlerTuple(ndivide=None)})
    legend.set_zorder(5)

def convergenceTemperature(ax, info, species):
    time, (dist,), folders = analysis.readFromVariation(
        folder = PARTICLE_VARIATION_FOLDER,
        dataset_names=[f"dist_fn/x_px/{species.value}"],
        processElement=lambda x: np.mean(x, axis=1),
        recursive=True
    )
    with h5py.File(next(PARTICLE_VARIATION_FOLDER.glob("*0032/rep_0.h5"))) as f:
        x_grid = f[f"Grid/x_px/{species.value}/X"][:]
        px_grid = f[f"Grid/x_px/{species.value}/Px"][:]

    with h5py.File(next(PARTICLE_VARIATION_FOLDER.glob("*8192/rep_0.h5"))) as f:
        p8192_time = f["Header/time"][:]
        p8192_dist = np.mean(f[f"/dist_fn/x_px/{species.value}"][:], axis=1)
        p8192_x_grid = f[f"Grid/x_px/{species.value}/X"][:]
        p8192_px_grid = f[f"Grid/x_px/{species.value}/Px"][:]

    time *= info.omega_pp
    p8192_time *= info.omega_pp
    # extract particle numbers
    particle_numbers = np.array([int(pfs[0].stem[-4:]) for pfs in folders])
    temperature = analysis.temperature1D(
        x_grid, px_grid, dist, info[species]
    )
    T_init = np.mean(temperature[:,:,:10], axis=-1)
    T_final = np.mean(temperature[:,:,-10:], axis=-1)
    T_diff = T_final - T_init
    mean_T_diff = np.mean(T_diff, axis=-1)
    std_T_diff = np.std(T_diff, axis=-1)

    p8192_temperature = analysis.temperature1D(
        p8192_x_grid, p8192_px_grid, p8192_dist, info[species]
    )
    p8192_T_diff = np.mean(p8192_temperature[-10:]) - np.mean(p8192_temperature[:10])

    l_vary = ax.errorbar(
        particle_numbers, mean_T_diff, yerr=std_T_diff,
        ls="", lw=1.5, color="white",
        marker="o", mec="black", mew=1.2, elinewidth=2, capsize=3, ecolor="black")
    _ = [cap.set(mew=2, mec="black") for cap in l_vary[1]]
    l_8192 = ax.plot(
        8192, p8192_T_diff, ls="", lw=1.5, color="white",
        marker="p", mec='black', mew=1.2)[-1]
    
    ax.set_ylabel(rf"Temperature $\Delta T_{{{species.symbol()},x}}$ (eV)")
    ax.legend(
        [(l_vary, l_8192)],
        [f"{species.name[0]}{species.name[1:].lower()}s (PIC)"], framealpha=1, 
        loc="center right", markerscale=0.7, edgecolor='black',
        handler_map={tuple: HandlerTuple(ndivide=None)})

def convergencePIC(info):
    fig, axes = plt.subplots(3,2, figsize=(6,6.5), constrained_layout=True, sharex=True)
    particle_numbers = convergenceFrequency(axes[0,0], info)
    convergenceWavenumber(axes[1,0], info)
    convergenceGrowthRate(axes[2,0], info)
    for ax, species in zip(axes[:,1], Species):
        convergenceTemperature(ax, info, species)
    
    for ax in axes[-1]:
        ax.set_xscale("log", base=2)
        ax.set(
            xlim=(0.5 * particle_numbers[0], 2 ** 14),
            xticks=2 ** np.linspace(4, 14, 6),
            xlabel=r"Macro-particles $N_\text{sim}\,/\,N_\text{cell}$",
        )
    for ax in axes[:,1]:
        ax.yaxis.set_label_position("right")   # move ylabel
        ax.yaxis.tick_right()                  # move tick labels to the right
        ax.tick_params(axis="y", which="both", left=True, right=True)
    for i, ax in enumerate(axes.T.flatten()):
        ax.text(0.05 if i < 3 else 0.18, 0.95,
            horizontalalignment='left',
            verticalalignment='top',
            s=rf"({chr(ord('a')+i)})",
            transform=ax.transAxes
        )

def vxVyDistSubplot(ax: plt.Axes, info: RunInfo, filename: Path, species: Species, time: float, xlim, ylim, colorbar, regimes, wave):
    f_v0 = _loadPxPyDistribution(info, species, filename, 0, True)[2]
    f_v_max = np.nanmax(f_v0)
    f_v_min = 2e-4# np.nanmin(f_v0[f_v0>0] / f_v_max)

    v_x, v_y, f_v = _loadPxPyDistribution(info, species, filename, time, True)
    dv_x = abs(v_x[1] - v_x[0])
    dv_y = abs(v_y[1] - v_y[0])
    v_x = np.concat([[v_x[0]-dv_x], v_x]) + dv_x / 2
    v_y = np.concat([[v_y[0]-dv_y], v_y]) + dv_y / 2
    f_v /= f_v_max
    f_v[f_v<=f_v_min] = f_v_min
    quad = ax.pcolormesh(v_x, v_y, f_v.T, norm="log", vmin=f_v_min, vmax=1.0, rasterized=True)

    u_alpha = int(filename.stem[-3:])
    ax.plot(
        0 if species == Species.PROTON else u_alpha * 1e3 / info.alpha.v_thermal, 0,
        marker="o", markeredgecolor="black", markeredgewidth=1, color="white")
    with h5py.File(filename) as f:
        sim_time = f['Header/time'][:] * info.omega_pp
        time_idx = np.argmin(np.abs(sim_time - time))
        Ex = f['Electric Field/ex'][:time_idx+1]
        Ey = f['Electric Field/ey'][:time_idx+1]
        x = f['Grid/grid/X'][:]
        y = f['Grid/grid/Y'][:]
        if x.ndim > 1:
            assert np.all(x == x[0]) and np.all(y == y[0])
            x = x[0]
            y = y[0]
    phi = np.percentile(np.abs(potentialFromElectricField(Ex[:-1], Ey[:-1], x, y)), 99.9)
    v_trap = np.sqrt(2 * info[species].si_charge * phi / info[species].si_mass) / info[species].v_thermal
    print(species, u_alpha, v_trap)

    with h5py.File(THEORY_U_ALPHA_FILE) as f:
        theory_v = f["u_alpha_bulk"][:] / 1e3
        theory_k = f["k_max"][:]
        theory_omega = f["omega_max"][:]
        theory_theta = f["theta_max"][:]

    v_ph = np.mean((theory_omega / theory_k)[theory_v>100]) / info[species].v_thermal
    theta = theory_theta[np.argmin(np.abs(theory_v - int(u_alpha)))]

    # NOTE: Test with wave angle from simulation (gives bad results - as expected from wave props figure).
    # if theta * 180 / np.pi > 20:
    #     assert np.sum(wave['u_alpha'] == u_alpha) == 1
    #     idx = int(np.where(wave['u_alpha'] == u_alpha)[0].squeeze())
    #     props = {k: v[idx] for k, v in wave.items()}
    #     wave_k = wave['k'][idx] / info.lambda_D
    #     wave_omega = wave['omega'][idx] * info.omega_pp
    #     wave_theta = wave['theta'][idx]
        
    #     v_ph = (wave_omega / wave_k) / info[species].v_thermal
    #     theta = wave_theta
    
    print(v_ph * info[species].v_thermal, v_trap * info[species].v_thermal)
    
    rect_pos = plt.Rectangle(
        xy=(
            (v_ph - v_trap) * np.cos(theta) + np.sin(theta) * (-4 * v_ph),
            (v_ph - v_trap) * np.sin(theta) - np.cos(theta) * (-4 * v_ph)
        ),
        width=100, height=2 * v_trap, angle=-(np.pi/2 - theta) * 180 / np.pi,
        edgecolor="black", zorder=1, facecolor="#c7c7c7", alpha=0.2
    )
    rect_neg = plt.Rectangle(
        xy=(
            +(v_ph + v_trap) * np.cos(theta) + np.sin(theta) * (-4 * v_ph),
            -(v_ph + v_trap) * np.sin(theta) + np.cos(theta) * (-4 * v_ph)
        ),
        width=100, height=2 * v_trap, angle=(np.pi/2 - theta) * 180 / np.pi,
        edgecolor="black", zorder=1, facecolor="#c7c7c7", alpha=0.2
    )
    if regimes:
        delta = 0 if species == Species.PROTON else 3 * v_trap
        alpha = theta
        if theta < np.pi / 180:
            delta = 1.2 * v_trap
        arrow_fix = 0.24 if species == Species.PROTON else 0.18
        ax.annotate(
            text='', xy=(
                (v_ph - v_trap - arrow_fix) * np.cos(theta) + np.sin(theta) * delta,
                (v_ph - v_trap - arrow_fix) * np.sin(theta) - np.cos(theta) * delta
            ),
            xytext=(
                (v_ph + v_trap + arrow_fix) * np.cos(theta) + np.sin(theta) * delta,
                (v_ph + v_trap + arrow_fix) * np.sin(theta) - np.cos(theta) * delta
            ), arrowprops=dict(arrowstyle='<->', lw=1.2, color="#000000")
        )
        ax.annotate(
            text='', xy=(
                +(v_ph - v_trap - arrow_fix) * np.cos(theta) + np.sin(theta) * delta,
                -(v_ph - v_trap - arrow_fix) * np.sin(theta) + np.cos(theta) * delta
            ),
            xytext=(
                +(v_ph + v_trap + arrow_fix) * np.cos(theta) + np.sin(theta) * delta,
                -(v_ph + v_trap + arrow_fix) * np.sin(theta) + np.cos(theta) * delta
            ), arrowprops=dict(arrowstyle='<->', lw=1.2, color="#000000")
        )
        ax.add_patch(rect_pos)
        ax.add_patch(rect_neg)
    ax.set_facecolor(plt.get_cmap('viridis').get_under())
    ax.set(
        xlim = xlim,
        ylim = ylim,
        xticks=np.arange(xlim[0], xlim[1]+1, 3),
        yticks=np.arange(ylim[0], ylim[1]+1, 3 if ylim[0] % 2 == 0 else 2),
    )
    ax.set_aspect("equal")
    return quad

def distributionsVelocitySpace(info, wave):
    file095 = next(V_FLOW_VARIATION_FOLDER.glob("*095.h5"))
    file140 = next(V_FLOW_VARIATION_FOLDER.glob("*140.h5"))
    fig, axes = plt.subplots(
        2, 3, sharey="row", constrained_layout=True,
        figsize=(6.5, 4.9),
        height_ratios=[10/12, 1.0],
    )
    
    i = 0
    for ax_row, species, xlim, ylim in zip(axes, [Species.PROTON, Species.ALPHA], [(-3, 9), (0, 9)], [(-6, 6), (-5, 5)]):
        for ax, filename, time, colorbar in zip(
            ax_row,
            [file095, file140, file140],
            [55.0, 55.0, 150.0],
            [True, True, False]):
            quad = vxVyDistSubplot(ax, info, filename, species, time, xlim, ylim, colorbar, colorbar, wave)
            ax.set_xlabel(f"Velocity $v_{{{species.symbol()},x}}\\,/\\,v^{{t=0}}_{{\\text{{t}}{species.symbol()}}}$")
            ax.text(
                0.03, 0.96,
                s=rf"$\mathbf{{({chr(ord('a')+i)})}}\,\,t\,\omega_\text{{pp}}={time:.0f}$",
                horizontalalignment="left",
                verticalalignment="top",
                color="white",
                transform=ax.transAxes,
            )
            i += 1
    for ax, species in zip(axes[:,0], [Species.PROTON, Species.ALPHA]):
        ax.set(
            ylabel=f"Velocity $v_{{{species.symbol()},y}}\\,/\\,v^{{t=0}}_{{\\text{{t}}{species.symbol()}}}$",
        )
    for ax, s_name, species in zip(axes[:,-1], ['Proton', 'Alpha'], [Species.PROTON, Species.ALPHA]):
        fig.colorbar(
            quad, ax=ax, location='right',
            label=rf"{s_name} dist. $\langle f_{species.symbol()}\rangle_\mathbf{{r}}$ (a.u.)",
            fraction=0.15, aspect=15, pad=0.03,
            shrink=0.78 if species == Species.PROTON else 0.73
        )
    fig.set_constrained_layout_pads(h_pad=0, w_pad=0, hspace=-0.3)
    fig.legend(
        [
            Line2D([0], [0], ls='', color='black', marker='$⟷$', markersize=10),
            Line2D([0], [0], ls='', color='white', marker='o', mec='black',mew =1, markersize=10),
            Patch(facecolor="#c7c7c7", alpha=0.5),
        ],
        [r"Transport $\partial f_s\,/\,\partial t$", r"Flow velocity $u_s^{t=0}$", 'Interaction range'],
        ncols=3, labelspacing=0.2, loc=(0.15,0.48), columnspacing=0.6, borderaxespad=0.8, framealpha=1,
        edgecolor='black', handletextpad=0.3, handlelength=1.5, handleheight=0.9)
    patch = Rectangle([0.34, 0], 0.006, 1.0, hatch='/////', ls='', facecolor='white',)
    fig.text(0.29, 1, r"$u_\alpha^{t=0}=95\,$km$\,/\,$s", ha='right', va='top')
    fig.text(0.52, 1, r"$u_\alpha^{t=0}=140\,$km$\,/\,$s", ha='left', va='top')
    fig.add_artist(patch)

if __name__ == "__main__":
    plt.style.use(MPLSTYLE_FILE)
    matplotlib.rcParams['figure.dpi'] = 100
    save = True
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
            bulk_velocity=1.0e5
        )
    )
    # Print information about the wave
    u_crit = 99.12905164474424
    with h5py.File(THEORY_U_ALPHA_FILE) as f:
        u_alpha = f["u_alpha_bulk"][:] * 1e-3
        gamma = f["gamma_max"][:] / info.omega_pp
        theta = f["theta_max"][:] * 180 / np.pi
        k_vec = f["k_max"][:]
        omega = f["omega_max"][:]
    v_ph = 1e-3 * np.mean((omega / k_vec)[u_alpha>u_crit])
    print("Xi         :", (u_crit - v_ph) * 1e3 / info.alpha.v_thermal)
    print("k (1/m)    :", np.mean(k_vec[u_alpha>u_crit]))
    print("omega (1/s):", np.mean(omega[u_alpha>u_crit]))
    print("v_ph / c_s :", v_ph * 1e3 / info.ionSoundSpeedElectronProton())
    # Print information about the energies
    deltaU = lambda T, s_info: (3/2) * s_info.number_density * constants.electron_volt * (np.mean(T[-10:]) - np.mean(T[:10]))
    for file in PIC_FILES_PREV:
        with h5py.File(file) as f:
            p_x = np.mean(f['Derived/Particles_Average_Px/Alphas'], axis=(1,2))
            p_y = np.mean(f['Derived/Particles_Average_Py/Alphas'], axis=(1,2))
            T_e = physics.kelvinToElectronVolt(
                np.mean(f[f"Derived/Temperature/Electrons"], axis=(1,2))
            )
            T_p = physics.kelvinToElectronVolt(
                np.mean(f[f"Derived/Temperature/Protons"], axis=(1,2))
            )
            T_a = physics.kelvinToElectronVolt(
                np.mean(f[f"Derived/Temperature/Alphas"], axis=(1,2))
            )
            u_a = int(file.stem[-3:]) * 1e3
            K_a = info.alpha.si_mass * info.alpha.number_density * u_a ** 2 / 2
            W = p_x ** 2 + p_y ** 2
            W /= np.max(W)
            W_loss = np.mean(W[:10]) - np.mean(W[-10:])
            delta_U_e = deltaU(T_e, info.electron)
            delta_U_p = deltaU(T_p, info.proton)
            delta_U_a = deltaU(T_a, info.alpha)
            delta_U_tot = delta_U_e + delta_U_p + delta_U_a
            print("u_alpha:", file.stem[-3:])
            print("Delta U_e     / Delta K_a:", delta_U_e / (W_loss * K_a))
            print("Delta U_p     / Delta K_a:", delta_U_p / (W_loss * K_a))
            print("Delta U_a     / Delta K_a:", delta_U_a / (W_loss * K_a))
            print("Delta U_total / Delta K_a:", delta_U_tot / (W_loss * K_a))
            print()
    # Produce plots for the publication
    folder = Path("figures/pub_rep")
    folder.mkdir(exist_ok=True, parents=True)
    wave = analysis.extractWaveProperties(info, PIC_FILES)
    temperatureElectricFieldMosaic(info)
    plt.savefig(folder / "electric_field_and_temperature_mosaic_t_tilde.pdf", dpi=200, bbox_inches='tight')
    flowVelocityWavePropsMosaic(info, wave)
    plt.savefig(folder / "flow_velocity_wave_props_mosaic.pdf", bbox_inches='tight', dpi=200)
    geometryParticleTransport(info)
    plt.savefig(folder / "geometry_velocity_space.pdf", bbox_inches="tight", dpi=200)
    flowVelocityEnergyTransferMosaic(info)
    plt.savefig(folder / "flow_velocity_energy_transfer_mosaic.pdf", bbox_inches='tight', dpi=200)
    convergencePIC(info)
    plt.savefig(folder / "convergence_temperature_wave_props.pdf", bbox_inches='tight', dpi=200)
    distributionsVelocitySpace(info, wave)
    plt.savefig(folder / "distributions_velocity_space.pdf", bbox_inches='tight', dpi=200)