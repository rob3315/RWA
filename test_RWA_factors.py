import unittest
import RWA_factors
import numpy as np

class TestRWA_factors(unittest.TestCase):

    def test_mul(self):
        lst_operatorType=[RWA_factors.A,RWA_factors.B,RWA_factors.C,RWA_factors.D]
        i1,j1,i2,j2=np.random.randint(-5,5,4)
        #i1,j1,i2,j2=1,0,0,1
        for op1 in lst_operatorType:
            X=op1((i1,j1))
            for op2 in lst_operatorType:
                Y=op2((i2,j2))
                E_1,E_2=np.random.random(2)
                self.assertAlmostEqual(np.linalg.norm(np.dot(X.get_value(E_1,E_2),Y.get_value(E_1,E_2))-(X*Y).get_value(E_1,E_2)),0)
    def test_scalmult(self):
        lst_operatorType=[RWA_factors.A,RWA_factors.B,RWA_factors.C,RWA_factors.D]
        i1,j1=np.random.randint(-5,5,2)
        for op1 in lst_operatorType:
            lamb=np.random.random()+1j*np.random.random()
            X=op1((i1,j1))
            E_1,E_2=np.random.random(2)
            c1=(lamb*X).get_value(E_1,E_2)
            c2=lamb*(X.get_value(E_1,E_2))
            self.assertAlmostEqual(np.linalg.norm(c1-c2),0)
    def test_add_num(self):
        lst_operatorType=[RWA_factors.A,RWA_factors.B,RWA_factors.C,RWA_factors.D]
        i1,j1=np.random.randint(-5,5,2)
        for op1 in lst_operatorType:
            lamb=np.random.random()+1j*np.random.random()
            X=op1((i1,j1))
            Y=op1((i1,j1))
            E_1,E_2=np.random.random(2)
            c1=X.get_value(E_1,E_2)+lamb*(Y.get_value(E_1,E_2))
            c2=(X+lamb*Y).get_value(E_1,E_2)
            self.assertAlmostEqual(np.linalg.norm(c1-c2),0)
    def test_primi(self):
        lst_operatorType=[RWA_factors.A,RWA_factors.B,RWA_factors.C,RWA_factors.D]
        #i,j=np.random.randint(-5,5,2)
        i,j=1,0        
        eps=1e-7
        for op1 in lst_operatorType:
            X=op1((i,j))
            Y=X.primi()
            E_1,E_2=np.random.random(2)
            c1=(Y.get_value(E_1+eps,0)-Y.get_value(E_1,0))/(eps)
            c2=X.get_value(E_1,E_2)
            #print(str(X),str(Y),c1,c2)
            self.assertAlmostEqual(np.linalg.norm(c1-c2),0,places=5)


if __name__=='__main__':
    unittest.main()