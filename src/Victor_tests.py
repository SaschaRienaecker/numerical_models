#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 15:21:13 2022

@author: victor
"""
#-----------------------------------------------------------------------------------------
# "Computing the ellipse"
k = omega_b/c



def ellipse(k, theta, R):
    kpar = k*np.cos(theta_res)
    Omega_ce = Omega_ce0 * R0/R
    
    vpar_bar = omega_b*kpar*light_speed**2/((kpar*c)**2 + (harmonic*Omega_ce))
    
    Delta_vpar = light_speed*np.sqrt(((kpar*light_speed)*** + (n*Omega_ce)**2 - omega_b**2))*(n*Omega_ce) \
        /((kpar*light_speed)**2+(n*Omega_ce)**2)
        
    Delta_vperp = light_speed*np.sqrt(((kpar*light_speed)**2 + (n*Omega_ce)**2 - omega_b**2) \
                                      /((kpar*light_speed)**2 + (n*Omega_ce)**2))
    
    vpar = np.linspace(-vmax,vmax,2*Nv)
    vperp = Delta_vperp*np.sqrt(1-((vpar-vpar_bar)/Delta_vpar)**2)
    # for i in range(2*Nv):
    #     vperp[i] = Delta_vperp*np.sqrt(1-((vpar[i]-vpar_bar)/Delta_vpar)**2)
    return vperp