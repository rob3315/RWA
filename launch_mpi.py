from mpi4py import MPI
from simu_mpi import *
import pickle
import sys

nproc_u=int(sys.argv[2])-1 #number of processeur processing
main=nproc_u # main proc number
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

alpha=float((sys.argv[1]))
leps1=[i*0.2 for i in range(7*5,10*5)]
leps2=[i*0.2 for i in range(7*5)]
dt=0.005
path='alpha{}.dict_{}'.format(alpha,dt)
l1=len(leps1)
l2=len(leps2)
l=l1*l2
if rank == main:
    dic={}
    for k in range(l):
        recvarray=comm.recv( source = k%nproc_u)
        dic[(2**(-leps1[k%l1]),2**(-leps2[k//l1]),'r')]=recvarray[0]
        dic[(2**(-leps1[k%l1]),2**(-leps2[k//l1]),'c')]=recvarray[1]
        with open(path, 'wb') as fp:
            pickle.dump(dic,fp,protocol=2)
else:
    i=rank
    while i<l:
        data=compute_err(alpha,2**(-leps1[i%l1]),2**(-leps2[i//l1]),dt)
        print('proc {} successfully completed task {}'.format(rank,i),data)
        comm.send(data, dest = main)
        i=i+nproc_u
    print("{:d} done".format(rank))
