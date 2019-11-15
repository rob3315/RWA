# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:37:07 2019

@author: robin
"""

from simu import *

def compute_err(alpha,eps1,eps2,dt):
    method='dopri5'
    inte=integrator(eps1,eps2,alpha)
    #real part
    def F(t,x):
         return np.dot(inte.A(t),x)
    def jac(t, y):
        return inte.A(t)
    v0=np.array([0,0,1])
    r = ode(F, jac).set_integrator(method)
    r.set_initial_value(v0, 0)
    while r.successful() and r.t < inte.tf:
        r.integrate(min(r.t+dt,inte.tf))
    if r.successful():
        psiR=r.y
    else :
        raise Exception('simulation was not successful')
    #complex part
    def F(t,x):
         return np.dot(inte.B(t),x)
    def jac(t, y):
        return inte.B(t)
    v0=np.array([0,0,1])
    r = ode(F, jac).set_integrator(method)
    r.set_initial_value(v0, 0)
    while r.successful() and r.t < inte.tf:
        r.integrate(min(r.t+dt,inte.tf))
    if r.successful():
        psiC=r.y
    else :
        raise Exception('simulation was not successful')
    return (psiR,psiC)