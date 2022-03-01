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

class SimuData:
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


###### Deprecated ###########
def plot_simu(simu_name):
    """
    Deprecated function
    """

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

    # Plot the density, temperature and Power profiles
    fig0 = plt.figure(0,figsize=(7, 5))
    ax01 = fig0.add_subplot(311)
    ax01.plot(vec_R, vec_Ne)
    ax01.set_ylabel("$N_e$ [$m^{-3}$]", fontsize = 20)
    ax02 = fig0.add_subplot(312)
    ax02.plot(vec_R, vec_Te / (1.602 * 10**(-19)))
    ax02.set_ylabel("$T_e$ [eV]", fontsize = 20)
    ax03 = fig0.add_subplot(313)
    ax03.plot(vec_R, (vec_Power[-1] - vec_Power)/vec_Power[-1], '-b')
    ax03.plot(vec_R, (vec_Albajar[-1] - vec_Albajar)/vec_Albajar[-1], '--r')
    ax03.set_xlabel("$R - R_0$", fontsize = 20)
    ax03.set_ylabel("$P_{abs}/P_{in}$", fontsize = 20)
    ax03.legend(["simulation","theory"])
    fig0.show()


    # Compute the position of maximum absorption
    dP_on_dR = np.diff(vec_Power)
    iR_max = np.argmax(dP_on_dR)
    fig1 = plt.figure(1,figsize=(7, 5))
    ax1 = fig1.add_subplot(111)
    plt.pcolor(Vpar, Vperp, np.transpose(Dn[iR_max,:,:]))
    ax1.set_xlabel("$v_{\parallel}$", fontsize = 20)
    ax1.set_ylabel("$v_{\perp}$", fontsize = 20)
    ax1.set_title("$D_{n}/(v_{Te}^2 \Omega_{ce})$", fontsize = 20)
    ax1.set_aspect('equal','box')
    plt.colorbar()

    fig1.show()

    print('P_{abs,tot}^{mod} / P_{abs,tot}^{ana}', (vec_Power[-1] - vec_Power[1])/(vec_Albajar[-1]-vec_Albajar[1]))

    saving = input("Do you want to save the figures? [y/n] (default = n)")
    if saving == ("y"):
        fig0.savefig(figpath / "Radial_profiles.pdf")
        fig1.savefig(figpath / "Dn_max.pdf")

def plot_profiles(simu_name, axs=None):

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
    ax03.plot(vec_R, (vec_Albajar[-1] - vec_Albajar)/vec_Albajar[-1], '--r')
    ax03.set_xlabel("$R - R_0$", fontsize = 20)
    ax03.set_ylabel("$P_{abs}/P_{in}$", fontsize = 20)
    ax03.legend(["simulation","theory"])

def plot_max_abs(simu_name, ax=None):

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
    im = plt.pcolor(Vpar, Vperp, np.transpose(Dn[iR_max,:,:]))
    ax.set_xlabel("$v_{\parallel}$", fontsize = 20)
    ax.set_ylabel("$v_{\perp}$", fontsize = 20)
    ax.set_title("$D_{n}/(v_{Te}^2 \Omega_{ce})$", fontsize = 20)
    ax.set_aspect('equal','box')
    fig.colorbar(im, ax=ax)

    return im
