# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:23:25 2019

@author: robin
"""
from scipy.integrate import ode
import numpy as np

def get_H(eps1,eps2,a,varphi,E,alpha):
    """a and varphi are function from [0,2pi]-> R"""
    def u(t):
        return eps1*a(eps1*eps2*t)*np.cos(2*E*t+varphi(eps1*eps2*t)/(eps1*eps2))
    return lambda t : np.array([[E+alpha, u(t)],[u(t),-E-alpha]],dtype=np.complex64)

a = lambda t : (1-np.cos(t))
u2 = lambda t : -3 *np.cos(t/2)
varphi= lambda t: -6 * np.sin(t/2) # primitive of u2 such as varphi(0)=0
E=2
#admissible alpha : (-1.5,1.5)
n=5#(=1/eps1)
eps1=1./n
eps2=eps1**2

alpha=1

psi0=np.array([0,1],dtype=np.complex64)#down

tf = 2*np.pi/(eps1*eps2)
dt = 0.001

nbstep=(int(tf/dt))
psi=np.copy(psi0)
print('shopae',nbstep,psi.shape)

H=get_H(eps1,eps2,a,varphi,E,alpha)
for step in range(nbstep-1):
    psi=psi-dt*1j*np.dot(H(step*dt),psi)
    psi=psi/np.linalg.norm(psi)
    print(step,nbstep,np.abs(psi[0]))