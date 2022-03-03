import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, m_e, epsilon_0, k, pi, c
from pathlib import Path

sys.path.append('../')
sys.path.append('../src')
sys.path.append('../utils')

from src.source import Simu
from src.plots import plot_max_abs, plot_profiles, SimuData, plot_Dn_over_ellipse
from utils.plotting import set_size, annotate_subplots
from src.mpi_job import frequency_scan, density_scan, temp_scan, angle_scan

figp = Path('../figures')

def n_crit(omega_b):
    return omega_b**2 * m_e * epsilon_0 / e**2

def f_ce(B0, R0, R):
    return e * B0 / m_e * R0 / R / 2 / np.pi

def R_res(harmonic, B0, R0, omega_b, gamma=1):
    return harmonic * e * B0 / m_e * R0 / gamma / omega_b

def gamma(v):
    return 1/(1 - (v/c)**2)**0.5

def vthe(T_keV):
    return (1e3 * e * T_keV / m_e)**0.5


def example_profiles():
    plt.style.use('../utils/tex.mplstyle')
    fs = set_size(width='article', fraction=1, aspect_r=1)

    simu_name = 'example'
    fig, axs = plt.subplots(2,1, sharex=True, figsize=fs)
    simu = Simu.load_pickle(simu_name)
    plot_profiles(simu_name, axs=axs)

    [ax1, ax2] = axs

    ax1.legend(frameon=True, loc='lower left', handlelength=0.5, ncol=1, framealpha=0.8)
    #xlim = ax1.get_xlim()
    #ax1.axhline(y=2 * , xmin=R_res_norm / (xlim[1] - xlim[0]))

    R_res_norm = (R_res(simu.harmonic, simu.B0, simu.R0, simu.omega_b, gamma=1) - simu.R0) / simu.a0
    #l = ax2.axvline(R_res_norm, ls='-', alpha=0.5)

    vec_Power = simu.vec_Power
    vec_R  = simu.vec_R
    R_norm = (vec_R - simu.R0) / simu.a0
    vec_Albajar = simu.vec_Albajar

    #ax2.set_xlim(right=0.7)

    # plot the slice as insetn0_cart
    extent = [-0.05, 0.05]
    axins = ax2.inset_axes([.05, .2, 0.4, 0.5], transform=ax2.transAxes)
    axins.plot(R_norm, (vec_Power[-1] - vec_Power)/vec_Power[-1], '-b')
    axins.plot(R_norm, (vec_Albajar[-1] - vec_Albajar)/vec_Albajar[-1], '--r')
    axins.set_xlim(extent)
    l = axins.axvline(R_res_norm, alpha=0.5)

    axins.text(R_res_norm, 1.1, '$R=R_\mathrm{res}$', ha='center', va='bottom', color=l.get_color())

    #axins.set_ylim()
    #ax2.indicate_inset_zoom(axins, edgecolor="black", alpha=0.5)

    ax2.legend(["simulation","theory"], loc='upper right', handlelength=1)

    annotate_subplots(axs, vpos=1.15)
    plt.tight_layout()

    fig.savefig(figp / 'example_profiles.pdf')
    return axs

def freq_variation():

    plt.style.use('../utils/tex.mplstyle')
    fs = set_size(width='article', fraction=1)

    # frequency scan
    simus = frequency_scan()

    fig, ax = plt.subplots(figsize=fs)

    for i in range(len(simus)):

        simu = Simu.load_pickle(simus[i].name)
        f = simu.omega_b * 1e-9 / 2 / np.pi # frequency in GHz
        f_label = '{:.0f}'.format(f)

        vec_R  = simu.vec_R
        vec_Ne = simu.vec_Ne
        vec_Te = simu.vec_Te
        vec_Power = simu.vec_Power
        vec_Albajar = simu.vec_Albajar

        R_norm = (vec_R - simu.R0) / simu.a0
        l, = ax.plot(R_norm , (vec_Power[-1] - vec_Power)/vec_Power[-1], '-', label=f_label)
        ax.plot(R_norm, (vec_Albajar[-1] - vec_Albajar)/vec_Albajar[-1], '--', color=l.get_color())

        R_res_norm = (R_res(simu.harmonic, simu.B0, simu.R0, simu.omega_b, gamma=1) - simu.R0) / simu.a0
        #ax.axvline(R_res_norm, color=l.get_color(), alpha=0.5, ymin=0, ymax=.3)
        ax.set_xlabel("$(R - R_0) / a$")
        ax.set_ylabel("$P_\mathrm{abs}/P_\mathrm{in}$")


        ax.set_xlim(right=1.7)
        ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        #ax.text()
        plt.tight_layout()

    ax.plot(R_norm, vec_Ne / vec_Ne.max(), alpha=0.4, ls='--')
    ax.legend(title='$f$ [GHz]', loc='upper right', handlelength=1.0)

    fig.savefig(figp / 'freq_variation.pdf')
    return ax

def density():
    plt.style.use('../utils/tex.mplstyle')
    fs = set_size(width='article')

    simus = density_scan()

    fig, ax = plt.subplots(figsize=fs)

    for i in range(len(simus)):
        simu = Simu.load_pickle(simus[i].name)
        n = simus[i].Ne0
        n_label = '{:.2f}'.format(n * 1e-19)

        vec_R  = simu.vec_R
        vec_Ne = simu.vec_Ne
        vec_Te = simu.vec_Te
        vec_Power = simu.vec_Power
        vec_Albajar = simu.vec_Albajar
        R_norm = (vec_R - simu.R0) / simu.a0
        print(simu.Te0 / 1e3 / e)

        l, = ax.plot(R_norm, (vec_Power[-1] - vec_Power)/vec_Power[-1], '-', label=n_label)
        ax.plot(R_norm, (vec_Albajar[-1] - vec_Albajar)/vec_Albajar[-1], '--', color=l.get_color())
        ax.set_xlabel("$(R - R_0)/a$")
        ax.set_ylabel("$P_\mathrm{abs}/P_\mathrm{in}$")
        #ax.legend(["simulation","theory"])
        ax.legend(title='$n_{e_0}$ [$10^{19}\mathrm{m}^{-3}$] =', frameon=False)
        ax.set_xlim(-0.3, 0.3)

        # annot = 'X-mode\n'
        # annot += r'$\theta_\mathrm{in}=\pi/2$'
        # ax.text(0.05, 0.05, annot, ha='left', va='bottom', transform=ax.transAxes)

        #ax.text()
        plt.tight_layout()
    fig.savefig(figp / 'density_scan.pdf')
    return ax
