 #-*- coding: utf-8 -*-
"""
@author: Peter Donnel
"""

# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# path to data directory
datap = Path('../data')


def testing_theta0(simu_name):
    # path to simulation directory
    simup = Path(datap / simu_name)

    # path to figures directory
    figpath = Path('../figures')

    # Load arrays for their exploitation
    vec_theta0  = np.load(simup / 'vec_theta0.npy')

    print(vec_theta0[0])
    print(vec_theta0)
    print(np.linalg.norm(vec_theta0 - vec_theta0[0]))

def plot_Dn_over_ellipse(simu_name):

    # path to simulation directory
    simup = Path(datap / simu_name)

    # path to figures directory
    figpath = Path('../figures')

    # Load arrays for their exploitation
    vec_R = np.load(simup / 'vec_R.npy')
    vec_Ne = np.load(simup / 'vec_Ne.npy')
    vec_Te = np.load(simup / 'vec_Te.npy')
    Vpar = np.load(simup / 'Vpar.npy')
    Vperp = np.load(simup / 'Vperp.npy')
    vec_Power = np.load(simup / 'vec_Power.npy')
    vec_Albajar = np.load(simup / 'vec_Albajar.npy')
    Dn = np.load(simup / 'Dn.npy')
    ellipse_vperp = np.load(simup / 'ellipse_vperp.npy')
    ellipse_vpar  = np.load(simup / 'ellipse_vpar.npy')

    X, Y = np.meshgrid(Vpar, Vperp)

    #Compute the position of maximum absorption and compare it to the ellipse
    dP_on_dR = np.diff(vec_Power)
    iR_max = np.argmax(dP_on_dR)
    # print("iR_max", iR_max)
    Z = np.transpose(Dn[iR_max,:,:])

    # Plotting the resonant diffusion coefficient
    fig, ax0 = plt.subplots()
    im = ax0.contourf(X, Y, Z, cmap = 'hot_r')
    fig.colorbar(im, ax=ax0)
    ax0.set_xlabel("$v_{\parallel} / v_{\mathrm{th}_e}$", fontsize = 20)
    ax0.set_ylabel("$v_{\perp} / v_{\mathrm{th}_e}$", fontsize = 20)
    ax0.set_title("$D_{n}/(v_{Te}^2 \Omega_{ce})$", fontsize = 20)

    # Plotting the ellipses to compare with
    ax0.plot(ellipse_vpar[0, iR_max, :], ellipse_vperp[0, iR_max, :], '--', alpha = 0.5)
    ax0.plot(ellipse_vpar[1, iR_max, :], ellipse_vperp[1, iR_max, :], '-.r', linewidth = 0.5, alpha = 0.5, label = r'$\sigma$')
    ax0.plot(ellipse_vpar[2, iR_max, :], ellipse_vperp[2, iR_max, :], '-.r', linewidth = 0.5, alpha = 0.5)
    ax0.plot(ellipse_vpar[3, iR_max, :], ellipse_vperp[3, iR_max, :], '-.b', linewidth = 0.5, alpha = 0.5, label = r'$3\sigma$')
    ax0.plot(ellipse_vpar[4, iR_max, :], ellipse_vperp[4, iR_max, :], '-.b', linewidth = 0.5, alpha = 0.5)
    ax0.legend()
    plt.show()


    # saving = input("Do you want to save the figures? [y/n] (default = n)")
    # if saving == ("y"):
    #     fig1.savefig(figpath / "Dn_max.pdf")

def plot_profiles(simu_name, axs=None, show_analy=True):

    if axs is None:
        fig, axs = plt.subplots(3,1,sharex=True)

    simu = SimuData(simu_name)

    [ax01, ax02, ax03] = axs

    vec_R  = simu.vec_R
    vec_Ne = simu.vec_Ne
    vec_Te = simu.vec_Te
    vec_Power = simu.vec_Power
    vec_Albajar = simu.vec_Albajar

    # Plot the density, temperature and Power profiles
    ax01.plot(vec_R, vec_Ne)
    ax01.set_ylabel("$N_e$ [$m^{-3}$]", fontsize = 20)
    ax02.plot(vec_R, vec_Te / (1.602 * 10**(-19)))
    ax02.set_ylabel("$T_e$ [eV]", fontsize = 20)
    ax03.plot(vec_R, (vec_Power[-1] - vec_Power)/vec_Power[-1], '-b')
    if show_analy:
        ax03.plot(vec_R, (vec_Albajar[-1] - vec_Albajar)/vec_Albajar[-1], '--r')
    ax03.set_xlabel("$R - R_0$", fontsize = 20)
    ax03.set_ylabel("$P_{abs}/P_{in}$", fontsize = 20)
    ax03.legend(["simulation","theory"])

def plot_max_abs(simu_name, ax=None, labels=True):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    simu = SimuData(simu_name)

    Vpar  = simu.Vpar
    Vperp = simu.Vperp
    Dn = simu.Dn
    vec_Power = simu.vec_Power

    # Compute the position of maximum absorption
    dP_on_dR = np.diff(vec_Power)
    iR_max = np.argmax(dP_on_dR)
    #im = plt.pcolor(Vpar, Vperp, np.transpose(Dn[iR_max,:,:]), **imshowargs)
    z = np.transpose(Dn[iR_max,:,:])
    z[z<1e-3*z.max()] = np.nan
    im = ax.imshow(z,extent=[Vpar[0], Vpar[-1], Vperp[0], Vperp[-1]],
                   origin='lower', cmap='seismic', vmax=z.max(), vmin=-z.max(), aspect='auto')
    if labels:
        ax.set_xlabel("$v_{\parallel}$")
        ax.set_ylabel("$v_{\perp}$")
        ax.set_title("$D_{n}/(v_{Te}^2 \Omega_{ce})$")
    # ax.set_aspect('equal','box')
    #fig.colorbar(im, ax=ax)
    # fig.colorbar(im, ax=ax, ticks=np.linspace(0, z.max(), 5))

    return im


############################ DEPRECATED ##########################################
class SimuData:
    """
    DEPRECATED: To load data, use the load_pickle function of the Simu class defined in source.py
    """
    def __init__(self, simu_name):

        # path to simulation directory
        simup = Path(datap / simu_name)

        self.simu_name = simu_name

        # Load arrays for their exploitation
        self.vec_R = np.load(simup / 'vec_R.npy')
        self.vec_Ne = np.load(simup / 'vec_Ne.npy')
        self.vec_Te = np.load(simup / 'vec_Te.npy')
        self.Vpar = np.load(simup / 'Vpar.npy')
        self.Vperp = np.load(simup / 'Vperp.npy')
        self.vec_Power = np.load(simup / 'vec_Power.npy')
        self.vec_Albajar = np.load(simup / 'vec_Albajar.npy')
        self.Dn = np.load(simup / 'Dn.npy')
