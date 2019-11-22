# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:35:53 2019

@author: robin
"""
import numpy as np
class Abstract_factor():
    def __init__(self,C):
        self.C=np.array(C,dtype=np.int)
        self.type=''
    def __str__(self):
        return "{}({}E1+{}E_2)".format(self.type,self.C[0],self.C[1])
    def __mul__(self,other):
        pass
    def copy(self):
        table={'A':A,'B':B,'C':C,'D':D}
        return table[self.type](self.C)
    def __eq__(self,other):
        if isinstance(other,Abstract_factor):
            return (self.type==other.type and self.C[0]==other.C[0] and self.C[1]==other.C[1])
        else :
            raise Exception('unable to compare')
class A(Abstract_factor):
    def __init__(self,C):
        Abstract_factor.__init__(self,C)
        self.type='A'
    def primi(self):
        return(B(self.C))
    def __mul__(self,other):
        if isinstance(other,Abstract_factor):
            table={'A':C,'B':D,'C':A,'D':B}
            return table[other.type](self.C-other.C)

class B(Abstract_factor):
    def __init__(self,C):
        Abstract_factor.__init__(self,C)
        self.type='B'
    def primi(self):
        return(A(self.C))
    def __mul__(self,other):
        if isinstance(other,Abstract_factor):
            table={'A':D,'B':C,'C':B,'D':A}
            return table[other.type](self.C-other.C)
class C(Abstract_factor):
    def __init__(self,C):
        Abstract_factor.__init__(self,C)
        self.type='C'
    def primi(self):
        return(D(self.C))
    def __mul__(self,other):
        if isinstance(other,Abstract_factor):
            table={'A':A,'B':B,'C':C,'D':D}
            return table[other.type](self.C+other.C)
class D(Abstract_factor):
    def __init__(self,C):
        Abstract_factor.__init__(self,C)
        self.type='D'
    def primi(self):
        return(C(self.C))
    def __mul__(self,other):
        if isinstance(other,Abstract_factor):
            table={'A':B,'B':A,'C':D,'D':C}
            return table[other.type](self.C+other.C)
    
class Hamiltonian():
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
        if isinstance(other,Hamiltonian) and other.n==self.n and other.m==self.m:
            for i in range(self.n):
                for j in range(self.m):
                    for ee in other.coeff[i,j]:
                        self.add_elt(i,j,ee)
    def add_elt(self,i,j,elt):
        """add elt in eps1^i eps_2^jif i and j small enough"""
        if i<self.n and j<self.m:
            for ee in self.coeff[i,j]:
                if elt==ee:
                    return
            self.coeff[i,j].append(elt)
            #print('adding element'+str(elt))
    def add_left_mult(self,i,j,Y):
        """add to the current hamiltonian the elt of eps_1^i eps_2^j Y H"""
        H_new=Hamiltonian(self.n,self.m)
        for ii in range(self.n-i):
            for jj in range(self.m-j):
                for elt in self.coeff[ii,jj]:
                    H_new.add_elt(i+ii,j+jj,Y*elt)
        self.add(H_new)
    def add_right_mult(self,i,j,Y):
        """add to the current hamiltonian the elt of eps_1^i eps_2^j H Y"""
        H_new=Hamiltonian(self.n,self.m)
        for ii in range(self.n-i):
            for jj in range(self.m-j):
                for elt in self.coeff[ii,jj]:
                    #print(str(elt),str(Y))
                    H_new.add_elt(i+ii,j+jj,elt*Y)
        self.add(H_new)
    def apply_RWA(self,i,j,k):
        X=self.coeff[i,j][k] # the term we want to eliminate
        Y=X.primi() #we applie the change of variable (I+ieps_1/f_X Y)
        self.add_left_mult(i,j,Y)# the term in ieps_1/f_X Y H
        self.add_elt(i+1,j+1,Y.copy())# the slow term
        #print(self)
        del self.coeff[i,j][k]# we delete X
        #print(self)
        Z=Y.copy()
        for order in range(1,self.n//i):
            self.add_right_mult(i*order,j,Z) # the expension of (I+i eps/f_x Y)^-1
            Z=Z*X
    def clean_order(self,i,j):
        def oscillation(lst):
            """return the first element with E_2"""
            for k in range(len(lst)):
                if lst[k].C[1]!=0 :
                    return k
            return -1
        while oscillation(self.coeff[i,j])!=-1:
            k=oscillation(self.coeff[i,j])
            #print('cleaning term {}'.format(str(self.coeff[i,j][k])))
            self.freq_deleted.add((self.coeff[i,j][k].C[0],self.coeff[i,j][k].C[1]))
            self.apply_RWA(i,j,k)
            
            
            
        
        
        
        
        
n=35
H=Hamiltonian(n,2)
H.coeff[1,0]=[A((1,0)),A((0,1))]
for k in range(n):
    H.clean_order(k,0)
    print(k)
for k in range(n):
    H.clean_order(k,1)
print(H.freq_deleted)
#print(H)
        