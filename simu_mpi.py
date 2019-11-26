# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:37:07 2019

@author: robin
"""

from simu import *

def compute_err(alpha,eps1,eps2,dt):
    method='dopri5'
    order=[1,2,4]
    tocompute=[(get_get_RWA_H(o),'r{}'.format(o)) for o in order]
    tocompute.append((get_A,'r'))
    lst_psi=[]
    for elt in tocompute:
        H=Hamiltonian(elt[0],elt[1])
        inte=integrator(eps1,eps2,alpha,H,use_dictio=False)
        psi=inte.integrate(dt,method)
        lst_psi.append((elt[1],psi))
    return (lst_psi)