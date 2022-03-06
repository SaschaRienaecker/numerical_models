"""
Use multiple CPU cores to run simultaneously different simulation jobs.
You need to have mpi4py installed.
Run the script from the command line as follows (e.g. if you have 4 available cores):
mpiexec --use-hwthread-cpus -n 4 ipython -m mpi4py mpi_job.py
"""
import numpy as np
from mpi4py import MPI
from pathlib import Path
from source import Simu
from math import ceil

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
root = 0

def mpi_task(simus, root=0):

    N_simu = len(simus)
    is_root = rank==root

    # in case N_simu > size (number of cores),
    # we will send several maps to each core:
    n_pproc = np.zeros(size).astype(int) # maps per core
    n_left = N_simu # maps left
    for i, npp in enumerate(n_pproc):
        n_pproc[i] = ceil(n_left / (size-i))
        n_left -= n_pproc[i]

    # displacements along the list
    displ = [npp for npp in n_pproc]
    displ = np.cumsum(displ) - displ
    displ = np.append(displ,  displ[-1]+n_pproc[-1])

    if rank == root:
        print('Number of simulations: ', N_simu)
        print('number of simulations per core :', n_pproc)

    simus_local = simus[displ[rank]:displ[rank+1]]
    print('rank {}: simu: {}'.format(rank, [s.name for s in simus_local]))

    for simu in simus_local:
        simu.compute()

def frequency_scan():

    # frequency scan
    F = np.array([63,70,80,90,100])
    N_simu = len(F)
    Names = ['freq_{}GHz'.format(f) for f in F]
    simus = [None] * N_simu

    for i in range(N_simu):
        simus[i] = Simu(Names[i],
                        B0=1.4,
                        R0=1.0,
                        a0=0.25,
                        harmonic=2,
                        theta_in=np.pi/2,
                        omega_b=F[i] * 1e9 * 2 * np.pi,
                        W0=0.02,
                        Power_in=1,
                        vmax=4,
                        Nv=100,
                        Nr=200,
                        Ne0=2.0e19,
                        Te0=2.0e3 * 1.602e-19
                        )
    return simus

def density_scan():

    Ne0 = np.array([0.25, 0.5, 1., 2.]) # 1e19 m⁻³
    N_simu = len(Ne0)
    Names = ['Ne0_{:.1f}e19'.format(n) for n in Ne0]
    print(Names)
    simus = [None] * N_simu

    for i in range(N_simu):
        simus[i] = Simu(Names[i],
                        B0=1.4,
                        R0=1.0,
                        a0=0.25,
                        harmonic=2,
                        theta_in=np.pi/2,
                        omega_b=7.8e10 * 2 * np.pi,
                        W0=0.02,
                        Power_in=1,
                        vmax=4,
                        Nv=100,
                        Nr=200,
                        Ne0=Ne0[i] * 1e19,
                        Te0= 2.0e3 * 1.602e-19
                        )
    return simus

def temp_scan(perp=True, ne=1e19):
    # Te0 = np.arange(1,9) # in keV
    Te0 = np.array([1,3,5,7]) # in keV
    N_simu = len(Te0)

    if perp:
        Names = ['Te0_{:.2f}keV'.format(T) for T in Te0]
        # Names = ['Te0_{:.2f}keV_lowdens'.format(T) for T in Te0]
        theta = np.pi/2
    else:
        Names = ['Te0_{:.2f}keV_60deg'.format(T) for T in Te0]
        theta = np.pi/3

    simus = [None] * N_simu

    for i in range(N_simu):
        simus[i] = Simu(Names[i],
                        B0=1.4,
                        R0=1.0,
                        a0=0.25,
                        harmonic=2,
                        theta_in=theta,
                        omega_b=7.8e10 * 2 * np.pi,
                        W0=0.02,
                        Power_in=1,
                        vmax=4,
                        Nv=100,
                        Nr=200,
                        Ne0=ne,
                        Te0=Te0[i] * 1e3 * 1.602e-19
                        )
    return simus

def angle_scan():

    theta = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]) * np.pi/2 # rad
    # theta = np.linspace(np.pi/4, np.pi/3, 4, endpoint=False) # 1e19 m⁻³
    N_simu = len(theta)
    Names = ['theta_{:.2f}'.format(t) for t in theta]
    print(Names)
    simus = [None] * N_simu

    for i in range(N_simu):
        simus[i] = Simu(Names[i],
                        B0=1.4,
                        R0=1.0,
                        a0=0.25,
                        harmonic=2,
                        theta_in=theta[i],
                        omega_b=7.8e10 * 2 * np.pi,
                        W0=0.02,
                        Power_in=1,
                        vmax=4,
                        Nv=100,
                        Nr=200,
                        Ne0=1e19,
                        Te0=2.0e3 * 1.602e-19
                        )
    return simus

def compare_O_and_X_mode():

    modes = ['X', 'X', 'X', 'O', 'O', 'O']
    harmonics = np.array([2, 3, 4, 1, 2, 3])
    freq_factor = harmonics / 2

    N_simu = len(modes)
    Names = ['mode_compar_{}'.format(n) for n in range(N_simu)]
    print(Names)
    simus = [None] * N_simu



    for i in range(N_simu):
        simus[i] = Simu(Names[i],
                        B0=1.4,
                        R0=1.0,
                        a0=0.25,
                        harmonic=harmonics[i],
                        theta_in=np.pi/2,
                        omega_b=7.8e10 * 2 * np.pi * freq_factor[i],
                        W0=0.02,
                        Power_in=1,
                        vmax=4,
                        Nv=100,
                        Nr=200,
                        Ne0=1e19,
                        Te0=4 * 2.0e3 * 1.602e-19,
                        mode=modes[i]
                        )
    return simus

if __name__ == '__main__':

    simus = temp_scan(ne=1e19)
    mpi_task(simus)
