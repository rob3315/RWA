# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:45:34 2019

@author: robin
"""

from simu import *
from random import shuffle

method='dopri5'
dt=0.005
alpha=0.0
leps1=[i*0.2 for i in range(7*5)]
leps2=[i*0.2 for i in range(7*5)]
#sing=OnlyOne(dt,alpha)
#dic=sing.instance.val
#print(len(dic.keys()))
#for j in range(len(leps2)-1):
#    for i in range(len(leps1)-1):
#        print(j,i,dic[(2**(-leps1[i]),2**(-leps2[j]),'r')])
#for computation only
#shuffle(leps1)
#shuffle(leps2)

#leps2=[7.5,8]
#inte=integrator(2**(-0.2),2**(-6),nocomputation=False,force_computation=True)
#print(inte.complex(dt,method))
#inte.real(dt,method)
plot_err(leps1,leps2,dt,alpha,nocomputation=True,force_computation=False)
sing=OnlyOne(dt,alpha)
dic=sing.instance.val
print(len(dic.keys()))