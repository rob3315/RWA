import unittest
import RWA_factors
import numpy as np

class TestRWA_factors(unittest.TestCase):

    def test_mul(self):
        lst_operatorType=[RWA_factors.A,RWA_factors.B,RWA_factors.C,RWA_factors.D]
        #i1,j1,i2,j2=np.random.randint(-5,5,4)
        i1,j1,i2,j2=1,0,0,1
        for op1 in lst_operatorType:
            X=op1((i1,j1))
            for op2 in lst_operatorType:
                Y=op2((i2,j2))
                E_1,E_2=1,3#np.random.random(2)
                print(np.dot(X.get_value(E_1,E_2),Y.get_value(E_1,E_2)))
                print((X*Y).get_value(E_1,E_2))
                print(X,Y,X*Y)
                self.assertAlmostEqual(np.linalg.norm(np.dot(X.get_value(E_1,E_2),Y.get_value(E_1,E_2))-(X*Y).get_value(E_1,E_2)),0)


if __name__=='__main__':
    unittest.main()