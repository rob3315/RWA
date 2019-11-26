# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:22:42 2019

@author: robin
"""

import unittest
from simu import *
import numpy as np
dt=0.01
method='dopri5'
H4=Hamiltonian(get_get_RWA_H(4),'random')
H2=Hamiltonian(get_get_RWA_H(2),'random')
H=Hamiltonian(get_get_RWA_H(1),'random')
H8=Hamiltonian(get_get_RWA_H(8),'random')
Hr=Hamiltonian(get_A,'random')
inter=integrator(0.1,0.1,alpha,Hr,nocomputation=False,use_dictio=False)
inte=integrator(0.1,0.1,alpha,H,nocomputation=False,use_dictio=False)
inte2=integrator(0.1,0.1,alpha,H2,nocomputation=False,use_dictio=False)
inte4=integrator(0.1,0.1,alpha,H4,nocomputation=False,use_dictio=False)
inte8=integrator(0.1,0.1,alpha,H8,nocomputation=False,use_dictio=False)
print(inte.integrate(dt,method))
print(inte2.integrate(dt,method))
print(inte4.integrate(dt,method))
print(inte8.integrate(dt,method))
print(inter.integrate(dt,method))