# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:37:07 2019

@author: robin
"""

from simu import *

def compute_err(alpha,eps1,eps2,dt):
    method='dopri5'
    H_R=Hamiltonian(get_A,'r')
    H_RWA=Hamiltonian(get_C,'r8')
    H_C=Hamiltonian(get_B,'c')
    inte_R=integrator(eps1,eps2,alpha,H_R,use_dictio=False)
    inte_RWA=integrator(eps1,eps2,alpha,H_RWA,use_dictio=False)
    inte_C=integrator(eps1,eps2,alpha,H_C,use_dictio=False)
    psiR=inte_R.integrate(dt,method)
    psiRWA=inte_RWA.integrate(dt,method)
    psiC=inte_C.integrate(dt,method)
    return (psiR,psiRWA,psiC)