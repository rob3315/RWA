# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:45:34 2019

@author: robin
"""

from simu import *
method='dopri5'
dt=0.001
alpha=0
leps1=[i*0.5 for i in range(2)]
leps2=[i*0.5 for i in range(34)]
#leps2=[7.5,8]
#inte=integrator(2**(-0.2),2**(-6),nocomputation=False,force_computation=True)
#print(inte.complex(dt,method))
#inte.real(dt,method)
plot_err(leps1,leps2,dt,alpha,nocomputation=True,force_computation=False)