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
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
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
    pathi='res/alpha{}.dict_{}'
    patho='res/alpha{}.dict_{}'
    #pathi='dict_{}'.format(dt)
    #patho='dict_{}'.format(dt)
    class __OnlyOne:
        def __init__(self,dt,alpha):
            self.dt=dt
            self.alpha=alpha
            try:
                with open(OnlyOne.pathi.format(alpha,dt), 'rb') as fp:
                    self.val = pickle.load(fp)
                    print('opening '+OnlyOne.pathi.format(alpha,dt))
            except IOError:
                print('not existing file')
                self.val={}
    instance = None
    def __init__(self,dt,alpha):
        if not OnlyOne.instance:
            OnlyOne.instance = OnlyOne.__OnlyOne(dt,alpha)
        else:
            pass
    def add(self,key,a):
        self.update()
        print('key',key,'value',a)
        OnlyOne.instance.val[key]=a
        self.save()
    def update(self):
        try:
            with open(OnlyOne.pathi.format(self.instance.alpha,self.instance.dt), 'rb') as fp:
                OnlyOne.instance.val = pickle.load(fp)
        except:
            print('unable to update')
    def save(self):
        print('saving file')
        with open(OnlyOne.patho.format(self.instance.alpha,self.instance.dt), 'wb') as handle:
            pickle.dump(OnlyOne.instance.val, handle)

class integrator():
    def __init__(self,eps1,eps2,alpha,E=1,nocomputation=False,force_computation=False):
        self.eps1=eps1
        self.eps2=eps2
        self.nocomputation=nocomputation
        self.force_computation=force_computation
        a = lambda t : (1-np.cos(t))
        #u2 = lambda t : -1 *np.cos(t/2)
        varphi= lambda t: -2 * np.sin(t/2) # primitive of u2 such as varphi(0)=0
        self.tf = 2*np.pi/(eps1*eps2)
        self.A=get_A(eps1,eps2,a,varphi,E,alpha)
        self.B=get_B(eps1,eps2,a,varphi,E,alpha)
        self.H=get_H(eps1,eps2,a,varphi,E,alpha)
        self.alpha=alpha
    
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
        sigleton_dict=OnlyOne(dt,self.alpha)
        dic=sigleton_dict.instance.val
        if dic.has_key((eps1,eps2,'r')) and self.force_computation==False:
            psi=dic[(eps1,eps2,'r')]
        elif self.nocomputation:
            return np.array([0,0,0])
        else:
            def F(t,x):
                return np.dot(self.A(t),x)
            def jac(t, y):
                return self.A(t)
            v0=np.array([0,0,1])
            r = ode(F, jac).set_integrator(method)
            r.set_initial_value(v0, 0)
            while r.successful() and r.t < self.tf:
                r.integrate(min(r.t+dt,self.tf))
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
        sigleton_dict=OnlyOne(dt,self.alpha)
        dic=sigleton_dict.instance.val
        if dic.has_key((eps1,eps2,'c')) and self.force_computation==False:
            psi=dic[(eps1,eps2,'c')]
        elif self.nocomputation:
            return np.array([0,0,0])
        else:
            def F(t,x):
                return np.dot(self.B(t),x)
            def jac(t, y):
                return self.B(t)
            v0=np.array([0,0,1])
            r = ode(F, jac).set_integrator(method)
            r.set_initial_value(v0, 0)
            while r.successful() and r.t < self.tf:
                r.integrate(min(r.t+dt,self.tf))
            if r.successful():
                psi=r.y
                sigleton_dict.add((eps1,eps2,'c'),psi)
            else :
                raise Exception('simulation was not successful')
            #print(r.t, r.y)
        #print(r.successful())
        return psi
def plot_err(leps1,leps2,dt,alpha,nocomputation=False,force_computation=False):
    method='dopri5'
    X, Y = np.meshgrid(leps1, leps2)
    Z1=np.zeros(X.shape)
    Z2=np.zeros(X.shape)
    Z3=np.zeros(X.shape)
    Z4=np.zeros(X.shape)
    Z5=np.zeros(X.shape)
    Z6=np.zeros(X.shape)
    for j in range(len(leps2)-1):
        for i in range(len(leps1)-1):
            #print(2**(-leps1[i]),2**(-leps2[j]))
            print(leps1[i],leps2[j])
            inte=integrator(2**(-leps1[i]),2**(-leps2[j]),alpha,nocomputation=nocomputation,force_computation=force_computation)
            print(inte.complex(dt,method))
            Z1[j,i]=-np.log2(1+inte.complex(dt,method)[2])# wierd convention
            Z2[j,i]=-np.log2(np.abs(inte.complex(dt,method)[2]-inte.real(dt,method))[2])# wierd convention
            #Z2[j,i]=-np.log2(np.linalg.norm(inte.complex(dt,method)-inte.real(dt,method)))# wierd convention
            Z3[j,i]=-np.log2(1+inte.real(dt,method)[2])
            Z6[j,i]=-np.log2(np.abs(1-np.linalg.norm(inte.complex(dt,method)))+np.abs(1-np.linalg.norm(inte.real(dt,method))))
            Z4[j,i]=0 if (i==0 and j==0)else -np.log2(1+inte.real(dt,method)[2])/(leps1[i]+leps2[j])
            Z5[j,i]=0 if (i==0 and j==0)else -np.log2(1+inte.complex(dt,method)[2])/(leps1[i]+leps2[j])
    #print(X,Y,Z)
    fig, axs = plt.subplots(3, 2)
    z_min=0
    z_max=35
    #levels = matplotlib.ticker.MaxNLocator(nbins=15).tick_values(z_min, z_max)
    ax = axs[0, 0]
    #c = ax.contourf(X, Y, Z1, cmap='jet', levels=levels)
    c = ax.pcolormesh(X,Y,Z1, cmap='jet')#, vmin=z_min, vmax=z_max)
    line=ax.plot(leps1,leps1,'red',linestyle='--',label='eps2/eps1=1')
    first_legend = ax.legend(handles=line, loc='lower right')
    ax.set_title('adiabatic error')
    fig.colorbar(c, ax=ax)
    
    ax = axs[0, 1]
    ax.set(ylim=(0, np.max(leps1)))
    c = ax.pcolormesh(X,Y,Z2, cmap='jet', vmin=z_min, vmax=z_max)
    line=ax.plot(leps1,3*np.array(leps1),'red',linestyle=':',label='eps1^3=eps2',linewidth=3)
    first_legend = ax.legend(handles=line, loc='lower right')
    ax.set_title('RWA error')
    fig.colorbar(c, ax=ax)
    
    ax = axs[1, 0]
    c = ax.pcolormesh(X,Y,Z3, cmap='jet', vmin=z_min, vmax=z_max)
    ax.set(ylim=(0, np.max(leps1)))
    ax.set_title('Total error')
    fig.colorbar(c, ax=ax)
    line3=ax.plot(leps1,leps1,'red',linestyle='--',label='eps1/eps2=1')
    line4=ax.plot(leps1,3*np.array(leps1),'red',linestyle=':',label='eps1^2=eps2',linewidth=3)
    
    ax = axs[1, 1]
    c = ax.pcolormesh(X,Y,Z4, cmap='jet', vmin=-1, vmax=8)
    ax.set_title('alpha real convergence rate T^-alpha')
    fig.colorbar(c, ax=ax)
    
    ax = axs[2, 1]
    c = ax.pcolormesh(X,Y,Z5, cmap='jet', vmin=-1, vmax=8)
    ax.set_title('alpha complex convergence rate T^-alpha')
    fig.colorbar(c, ax=ax)
    
    ax = axs[2, 0]
    c = ax.pcolormesh(X,Y,Z6, cmap='jet')
    ax.set_title('numerical error')
    fig.colorbar(c, ax=ax)
    
    
    fig.tight_layout()
    plt.show()
    
#admissible alpha : (-0.5,0.5)
#method=['lsoda','vode','dopri5','dop853']
#print(inte.real_euler(dt))
#psi=inte.complex_euler(dt)
#print(np.abs(psi[0])**2-np.abs(psi[1])**2)
#for m in method:
#    print('method :'+m)
#    for dt in (1e-1,1e-2,1e-3,1e-4):
#        temp=inte.complex(dt,m)
#        print('dt : '+ str(dt),temp,1-np.linalg.norm(temp))
