# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:35:53 2019

@author: robin
"""
#import sympy
import numpy as np
import time
import sympy
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

class Abstract_factor():
    def __init__(self,coef_freq):
        self.coef_freq=np.array(coef_freq,dtype=np.int)
        self.type=''
        self.front_coeff=1
        self.table={'A':A,'B':B,'C':C,'D':D}
    def __str__(self):
        return "{}*{}({}E_1+{}E_2)".format(self.front_coeff,self.type,self.coef_freq[0],self.coef_freq[1])
    def __mul__(self,other):
        pass
    
    def __add__(self,other):
        if isinstance(other,Abstract_factor):
            if self.type==other.type and self.coef_freq[0]==other.coef_freq[0] and self.coef_freq[1]==other.coef_freq[1]:
                res=self.table[self.type](self.coef_freq)
                res.front_coeff=self.front_coeff+other.front_coeff
                return res
            else : raise Exception("cannot add {}+{}".format(self,other))
        else : raise Exception("unable to add non Abstract_factor")
    
    def __rmul__(self,other):
        """scalar multiplication"""
        res=(self.table[self.type])(self.coef_freq)
        res.front_coeff=other*self.front_coeff
        return res
    
    def copy(self):
        table={'A':A,'B':B,'C':C,'D':D}
        return self.front_coeff*table[self.type](self.coef_freq)
    def __eq__(self,other):
        if isinstance(other,Abstract_factor):
            return (self.type==other.type and self.coef_freq[0]==other.coef_freq[0] and self.coef_freq[1]==other.coef_freq[1])
        else :
            raise Exception('unable to compare')
    def shape(self): pass
    def get_value(self,VE_1,VE_2):
        E=VE_1*self.coef_freq[0]+VE_2*self.coef_freq[1]
        return self.front_coeff*self.shape()(E)

class A(Abstract_factor):
    def __init__(self,coef_freq):
        Abstract_factor.__init__(self,coef_freq)
        self.type='A'
    def primi(self):
        return(B(self.coef_freq))
    def __mul__(self,other):
        if isinstance(other,Abstract_factor):
            table={'A':C,'B':D,'C':A,'D':B}
            front_coeff={'A':1,'B':-1,'C':1,'D':-1}
            return self.front_coeff*other.front_coeff*front_coeff[other.type]*table[other.type](self.coef_freq-other.coef_freq)
    def shape(self):
        return lambda E: np.array([[0,np.exp(1j*E)],[np.exp(-1j*E),0]])

class B(Abstract_factor):
    def __init__(self,coef_freq):
        Abstract_factor.__init__(self,coef_freq)
        self.type='B'
    def primi(self):
        return(-1*A(self.coef_freq))
    def __mul__(self,other):
        if isinstance(other,Abstract_factor):
            table={'A':D,'B':C,'C':B,'D':A}
            front_coeff={'A':1,'B':1,'C':1,'D':1}
            return self.front_coeff*other.front_coeff*front_coeff[other.type]*table[other.type](self.coef_freq-other.coef_freq)
    def shape(self):
        return lambda E: np.array([[0,-1j*np.exp(1j*E)],[1j*np.exp(-1j*E),0]])

class C(Abstract_factor):
    def __init__(self,coef_freq):
        Abstract_factor.__init__(self,coef_freq)
        self.type='C'
    def primi(self):
        return(D(self.coef_freq))
    def __mul__(self,other):
        if isinstance(other,Abstract_factor):
            table={'A':A,'B':B,'C':C,'D':D}
            front_coeff={'A':1,'B':1,'C':1,'D':1}
            return self.front_coeff*other.front_coeff*front_coeff[other.type]*table[other.type](self.coef_freq+other.coef_freq)
    def shape(self):
        return lambda E: np.array([[np.exp(1j*E),0],[0,np.exp(-1j*E)]])
class D(Abstract_factor):
    def __init__(self,coef_freq):
        Abstract_factor.__init__(self,coef_freq)
        self.type='D'
    def primi(self):
        return(-1*C(self.coef_freq))
    def __mul__(self,other):
        if isinstance(other,Abstract_factor):
            table={'A':B,'B':A,'C':D,'D':C}
            front_coeff={'A':1,'B':-1,'C':1,'D':-1}
            return self.front_coeff*other.front_coeff*front_coeff[other.type]*table[other.type](self.coef_freq+other.coef_freq)
    def shape(self):
        return lambda E: np.array([[-1j*np.exp(1j*E),0],[0,1j*np.exp(-1j*E)]])
    
class Abstract_Hamiltonian():
    
    def __init__(self,n,m):
        self.n=n
        self.m=m
        self.coeff=np.empty((n,m), dtype=object)#2D array of list of Abstract_factor
        for i in range(self.n):
            for j in range(self.m):
                self.coeff[i,j]=[]
        self.freq_deleted=set()
    
    def __str__(self):
        string=''
        for j in range(self.m):
            for i in range(self.n):
                if len(self.coeff[i,j])>0:
                    string+='\n eps_1^{}eps_2^{} ('.format(i,j)
                    for elt in self.coeff[i,j]:
                        string+=str(elt)+'+'
                    string=string[:-1]+')'
        return string
    
    def add(self,other):
        assert isinstance(other,Abstract_Hamiltonian) and other.n==self.n and other.m==self.m
        for i in range(self.n):
            for j in range(self.m):
                for ee in other.coeff[i,j]:
                    self.add_elt(i,j,ee)
    
    def add_elt(self,i,j,elt):
        """add elt in eps1^i eps_2^jif i and j small enough"""
        if i<self.n and j<self.m:
            for k in range(len(self.coeff[i,j])):
                ee=self.coeff[i,j][k]
                if elt==ee:
                    ee=ee+elt
                    if ee.front_coeff==0:
                        del(self.coeff[i,j][k])
                    return
            self.coeff[i,j].append(elt)
            #print('adding element'+str(elt))
    
    def add_left_mult(self,i,j,Y):
        """add to the current Abstract_Hamiltonian the elt of eps_1^i eps_2^j Y H"""
        H_new=Abstract_Hamiltonian(self.n,self.m)
        for ii in range(self.n-i):
            for jj in range(self.m-j):
                for elt in self.coeff[ii,jj]:
                    H_new.add_elt(i+ii,j+jj,Y*elt)
        self.add(H_new)
    
    def right_mult(self,i,j,Y):
        """add to the current Abstract_Hamiltonian the elt of eps_1^i eps_2^j H Y"""
        H_new=Abstract_Hamiltonian(self.n,self.m)
        for ii in range(self.n-i):
            for jj in range(self.m-j):
                for elt in self.coeff[ii,jj]:
                    #print(str(elt),str(Y))
                    H_new.add_elt(i+ii,j+jj,elt*Y)
        return H_new
    
    def apply_RWA(self,i,j,k):
        X=self.coeff[i,j][k] # the term we want to eliminate
        Y=X.primi() #we applie the change of variable (I+ieps_1/f_X Y)
        if X.coef_freq[1]>=0:
            g_X=sympy.Symbol('g_{}{}'.format(X.coef_freq[0],X.coef_freq[1]))#g_ij =1/(if_1+jf_2)
        else:
            g_X=-sympy.Symbol('g_{}{}'.format(-X.coef_freq[0],-X.coef_freq[1]))#g_ij =1/(if_1+jf_2)
        self.add_left_mult(i,j,1j*g_X*Y)# the term in ieps_1/f_X Y H
        self.add_elt(i+1,j+1,-1*Y.copy())# the slow term TODO prefacteur
        self.add_elt(i,j,-1*X.copy())
        #print(self)
        #del self.coeff[i,j][k]# we delete X
        #print(self)
        Z=(-1j*g_X)*Y.copy()
        lstH=[]
        for order in range(1,self.n//i):
            lstH.append(self.right_mult(i*order,j,1./order *Z)) # the expension of (I+i eps/f_x Y)^-1
            Z=Z*(-1j*g_X*Y)
        for H in lstH:
            self.add(H)
    def clean_order(self,i,j):
        """clean the coefficientall s of frequency containing f_2 in front of eps_1^i eps_2^j"""
        def oscillation(lst):
            """return the first element with E_2"""
            for k in range(len(lst)):
                if lst[k].coef_freq[1]!=0 :
                    return k
            return -1
        while oscillation(self.coeff[i,j])!=-1:
            k=oscillation(self.coeff[i,j])
            #print('cleaning term {}'.format(str(self.coeff[i,j][k])))
            self.freq_deleted.add((self.coeff[i,j][k].coef_freq[0],self.coeff[i,j][k].coef_freq[1]))
            self.apply_RWA(i,j,k)
    
    def get_H(self,eps1,eps2,a,varphi,E,alpha,dvarphi):
        def H(t):
            x=E+alpha
            y=
            
            
        
        
        
        
if __name__ == "__main__":    
    n=10
    H=Abstract_Hamiltonian(n,2)
    H.coeff[1,0]=[A((1,0)),A((0,1))]
    print(H)
    #H.apply_RWA(1,0,1)
    for k in range(n):
        H.clean_order(k,0)
    print(H.freq_deleted)
    print(H)