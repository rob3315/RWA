# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:23:25 2019

@author: robin
"""
import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
import time
import pickle
dt=0.001
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print '%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000)
        return result
    return timed

def get_H(eps1,eps2,a,varphi,E,alpha):
    """a and varphi are function from [0,2pi]-> R"""
    def u(t):
        return eps1*a(eps1*eps2*t)*np.cos(2*E*t+varphi(eps1*eps2*t)/(eps1*eps2))
    return lambda t : np.array([[E+alpha, u(t)],[u(t),-E-alpha]],dtype=np.complex64)

def get_A(eps1,eps2,a,varphi,E,alpha):
    def u(t):
        return eps1*a(eps1*eps2*t)*np.cos(2*E*t+varphi(eps1*eps2*t)/(eps1*eps2))
    return lambda t: np.array([[0,-2*(E+alpha),0],
                [2*(E+alpha),0,-2*u(t)],
                [0,2*u(t),0]])
def get_B(eps1,eps2,a,varphi,E,alpha):
    def u1(t):
        return 0.5*eps1*a(eps1*eps2*t)*np.cos(2*E*t+varphi(eps1*eps2*t)/(eps1*eps2))
    def u2(t):
        return 0.5*eps1*a(eps1*eps2*t)*np.sin(2*E*t+varphi(eps1*eps2*t)/(eps1*eps2))
    return lambda t: np.array([[0,-2*(E+alpha),2*u2(t)],
                [2*(E+alpha),0,-2*u1(t)],
                [-2*u2(t),2*u1(t),0]])
	
class OnlyOne:
    """singleton, contain a dictionary with already computed simulation"""
    pathi='dict_{}'.format(dt)
    patho='dict_{}'.format(dt)
    class __OnlyOne:
        def __init__(self):
            try:
                with open(OnlyOne.pathi, 'rb') as fp:
                    self.val = pickle.load(fp)
            except IOError:
                print('not existing file')
                self.val={}
    instance = None
    def __init__(self):
        if not OnlyOne.instance:
            OnlyOne.instance = OnlyOne.__OnlyOne()
        else:
            pass
    def add(self,key,a):
        print('key',key,'value',a)
        OnlyOne.instance.val[key]=a
        self.save()
    def save(self):
        print('saving file')
        with open(OnlyOne.patho, 'wb') as handle:
            pickle.dump(OnlyOne.instance.val, handle)

class integrator():
    def __init__(self,eps1,eps2,alpha=0,E=1):
        self.eps1=eps1
        self.eps2=eps2
        a = lambda t : (1-np.cos(t))
        #u2 = lambda t : -3 *np.cos(t/2)
        varphi= lambda t: -6 * np.sin(t/2) # primitive of u2 such as varphi(0)=0
        self.tf = 2*np.pi/(eps1*eps2)
        self.A=get_A(eps1,eps2,a,varphi,E,alpha)
        self.B=get_B(eps1,eps2,a,varphi,E,alpha)
        self.H=get_H(eps1,eps2,a,varphi,E,alpha)
    
    @timeit
    def complex_euler(self,dt,verbose=False):
        psi0=np.array([1,0],dtype=np.complex64)
        nbstep=(int(self.tf/dt))
        psi=np.copy(psi0)
        for step in range(nbstep-1):
            psi=psi-dt*1j*np.dot(self.H(step*dt),psi)
            psi=psi/np.linalg.norm(psi)
            if verbose:
                print(step,nbstep,np.abs(psi[0])**2-np.abs(psi[1])**2)
        return psi
    @timeit
    def real_euler(self,dt,verbose=False):
        psi0=np.array([0,0,1])
        nbstep=(int(self.tf/dt))
        psi=np.copy(psi0)
        for step in range(nbstep-1):
            psi=psi+dt*np.dot(self.A(step*dt),psi)
            psi=psi/np.linalg.norm(psi)
            if verbose:
                print(step,nbstep,psi)
        return psi
    @timeit
    def real(self,dt,method):
        eps1=self.eps1
        eps2=self.eps2
        sigleton_dict=OnlyOne()
        dic=sigleton_dict.instance.val
        if dic.has_key((eps1,eps2,'r')):
            psi=dic[(eps1,eps2,'r')]
        else:
            def F(t,x):
                return np.dot(self.A(t),x)
            def jac(t, y):
                return self.A(t)
            v0=np.array([0,0,1])
            r = ode(F, jac).set_integrator(method)
            r.set_initial_value(v0, 0)
            while r.successful() and r.t < self.tf:
                r.integrate(r.t+dt)
                #print(r.t, r.y)
            #print(r.successful())
            if r.successful():
                psi=r.y
                sigleton_dict.add((eps1,eps2,'r'),psi)
            else :
                raise Exception('simulation was not successful')
        return psi
    @timeit
    def complex(self,dt,method):
        eps1=self.eps1
        eps2=self.eps2
        sigleton_dict=OnlyOne()
        dic=sigleton_dict.instance.val
        if dic.has_key((eps1,eps2,'c')):
            psi=dic[(eps1,eps2,'c')]
        else:
            def F(t,x):
                return np.dot(self.B(t),x)
            def jac(t, y):
                return self.B(t)
            v0=np.array([0,0,1])
            r = ode(F, jac).set_integrator(method)
            r.set_initial_value(v0, 0)
            while r.successful() and r.t < self.tf:
                r.integrate(r.t+dt)
            if r.successful():
                psi=r.y
                sigleton_dict.add((eps1,eps2,'c'),psi)
            else :
                raise Exception('simulation was not successful')
            #print(r.t, r.y)
        #print(r.successful())
        return psi
def plot_err():
    method='dopri5'
    leps1=[i for i in range(7)]
    leps2=[i for i in range(7)]
    X, Y = np.meshgrid(leps1, leps2)
    Z=np.zeros(X.shape)
    for i in range(len(leps1)-1):
        for j in range(len(leps2)-1):
            print(2**(-leps1[i]),2**(-leps2[j]))
            inte=integrator(2**(-leps1[i]),2**(-leps2[j]))
            #Z[j,i]=-np.log10(1+inte.complex(dt,method)[2])# wierd convention
            Z[j,i]=-np.log2(np.linalg.norm(inte.complex(dt,method)-inte.real(dt,method)))# wierd convention
            #Z[j,i]=-np.log10(1+inte.real(dt,method)[2])
    print(X,Y,Z)
    plt.pcolor(X, Y, Z)
    plt.colorbar()
    plt.show()
#admissible alpha : (-1.5,1.5)
n=2#(=1/eps1)
eps1=1./n
eps2=eps1**2
#inte=integrator(eps1,eps2)
a=OnlyOne()
a.save()
plot_err()
#method=['lsoda','vode','dopri5','dop853']
#print(inte.real_euler(dt))
#psi=inte.complex_euler(dt)
#print(np.abs(psi[0])**2-np.abs(psi[1])**2)
#for m in method:
#    print('method :'+m)
#    for dt in (1e-1,1e-2,1e-3,1e-4):
#        temp=inte.complex(dt,m)
#        print('dt : '+ str(dt),temp,1-np.linalg.norm(temp))
