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
    Omega_b = np.linspace(6, 10, 8) * 1e10 * 2 * np.pi
    N_simu = len(Omega_b)
    Names = ['freq_{}'.format(i) for i in range(len(Omega_b))]
    simus = [None] * N_simu

    for i in range(N_simu):
        simus[i] = Simu(    Names[i],
                        B0=1.4,
                        R0=1.0,
                        a0=0.25,
                        harmonic=2,
                        theta_in=np.pi/2,
                        omega_b=Omega_b[i],
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

    Ne0 = np.linspace(0.1, 2, 8) # 1e19 m⁻³
    N_simu = len(Ne0)
    Names = ['Ne0_{:.2f}e19'.format(n) for n in Ne0]
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
                        Te0=2.0e3 * 1.602e-19
                        )
    return simus

def temp_scan(perp=True):
    Te0 = np.arange(1,9) # in keV
    N_simu = len(Te0)

    if perp:
        Names = ['Te0_{:.2f}keV'.format(T) for T in Te0]
    else:
        Names = ['Te0_{:.2f}keV_60deg'.format(T) for T in Te0]

    simus = [None] * N_simu

    for i in range(N_simu):
        simus[i] = Simu(Names[i],
                        B0=1.4,
                        R0=1.0,
                        a0=0.25,
                        harmonic=2,
                        theta_in=np.pi/3,
                        omega_b=7.8e10 * 2 * np.pi,
                        W0=0.02,
                        Power_in=1,
                        vmax=4,
                        Nv=100,
                        Nr=200,
                        Ne0=1e19,
                        Te0=Te0[i] * 1e3 * 1.602e-19
                        )
    return simus


if __name__ == '__main__':

    simus = temp_scan()
    mpi_task(simus)
