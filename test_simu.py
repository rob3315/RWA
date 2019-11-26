# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:22:42 2019

@author: robin
"""

import unittest
from simu import *
import numpy as np
dt=0.01
alpha=0
method='dopri5'
H4=Hamiltonian(get_get_RWA_H(4),'random')
H2=Hamiltonian(get_get_RWA_H(2),'random')
H=Hamiltonian(get_get_RWA_H(1),'random')
H8=Hamiltonian(get_get_RWA_H(8),'random')
Hr=Hamiltonian(get_A,'random')
eps1,eps2=0.1,1
inter=integrator(eps1,eps2,alpha,Hr,nocomputation=False,use_dictio=False)
inte=integrator(eps1,eps2,alpha,H,nocomputation=False,use_dictio=False)
inte2=integrator(eps1,eps2,alpha,H2,nocomputation=False,use_dictio=False)
inte4=integrator(eps1,eps2,alpha,H4,nocomputation=False,use_dictio=False)
#inte8=integrator(eps1,eps2,alpha,H8,nocomputation=False,use_dictio=False)
""" ref=inter.integrate(dt,method)
print(np.linalg.norm(inte.integrate(dt,method)-ref))
print(np.linalg.norm(inte2.integrate(dt,method)-ref))
print(np.linalg.norm(inte4.integrate(dt,method)-ref))
#print(inte8.integrate(dt,method))
print(ref)"""
T= np.linspace(0,2*np.pi/(eps1*eps2),800)
for i in range(3):
    for j in range(3):
        CHam1=np.array([inte2.Ha(t)[i,j] for t in T])
        CHam2=np.array([inte4.Ha(t)[i,j] for t in T])
        #plt.plot(T,CHam1-CHam2) """
dvarphi = lambda t : -1 *np.cos(t/2)
a = lambda t : (1-np.cos(t))
varphi= lambda t: -2 * np.sin(t/2) # primitive of dvarphi such as varphi(0)=0
def P1(x): return 1 -0.5*x**2
def eps(t) : return eps1*a(eps1*eps2*t)/2
def g_01(t) : return 1./(4*1+2*alpha-dvarphi(eps1*eps2*t))
def u1(t):return eps(t)*P1(eps(t)*g_01(t))
plt.plot(eps(T)*(eps(T)*g_01(T))**2)
plt.show()