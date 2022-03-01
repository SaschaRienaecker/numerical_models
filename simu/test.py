"""
Simulation input file
"""

import numpy as np

# Plasma inputs
Ne0 = 1.0 * 10**(19)                  # maximum electron density [m^{-3}]
Te0 = 2.0 * 10**3 * 1.602 * 10**(-19) # maximum of electron temperature [J] (Warning: kB included)
B0 = 1.4                              # magnetic field on axis [T]
R0 = 1.0                              # major radius [m]
a0 = 0.25                             # minor radius [m]


# Beam inputs
harmonic = 2                          # harmonic of the cyclotron frequency
theta_in = np.pi/2                    # Toroidal angle of injection
omega_b = 7.8 * 10**10 * 2 * np.pi    # beam pulsation [Hz]
W0 = 0.02                             # beam width [m]
Power_in = 1                          # power input of the beam [W]


# Numerical imput data
vmax = 4   # maximal velocity (normalized to the local thermal velocity)
# Nv = 300   # number of grid points in velocity space (Nv for vperp, 2*Nv for vpar)
Nv = 100   # number of grid points in velocity space (Nv for vperp, 2*Nv for vpar)
# Nr = 400   # number of grid points for the major radius direction
Nr = 200   # number of grid points for the major radius direction


########### Deprecated ##########

# class Simu:
#     def __init__(name, B0, R0, a0, harmonic, theta_in, omega_b, W0, Power_in, vmax, Nv, Nr):
#         self.name = name
#
#         # Plasma inputs
#         self.B0      = B0                           # magnetic field on axis [T]
#         self.R0      = R0                           # major radius [m]
#         self.a0      = a0                           # minor radius [m]
#
#         # Beam inputs
#         self.harmonic   = harmonic                # harmonic of the cyclotron frequency
#         self.theta_in   = theta_in                # Toroidal angle of injection
#         self.omega_b    = omega_b                 # beam pulsation [Hz]
#         self.W0         = W0                      # beam width [m]
#         self.Power_in   = Power_in                # power input of the beam [W]
#
#         # Numerical imput data
#         self.vmax       = vmax   # maximal velocity (normalized to the local thermal velocity)
#         self.Nv         = Nv   # number of grid points in velocity space (Nv for vperp, 2*Nv for vpar)
#         self.Nr         = Nr   # number of grid points for the major radius direction


# testsim = Simu('test',
#                 B0=1.4,
#                 R0=1.0,
#                 a0=0.25,
#                 harmonic=2,
#                 theta_in=np.pi/2,
#                 omega_b=7.8 * 10**10 * 2 * np.pi,
#                 W0=0.02,
#                 Power_in=1,
#                 vmax=4,
#                 Nv=50,
#                 Nr=50
#                 )
