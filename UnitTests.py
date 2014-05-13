import scipy.io
import unittest
import numpy as np
from MatrixFactorization import ANOVARegress
from scipy.sparse import lil_matrix, csr_matrix
from DataLoading import readMSDData

basePath = 'C:/Users/William/Downloads/'

class UnitTests(unittest.TestCase):
    def testDataLoad(self):
        M = readMSDData()
        # Check if all values are between 0 and 1
        self.assertLessEqual(M.max(), 1, "Entries below 1")
        self.assertGreaterEqual(M.min(),0,"Entries above 0")
        MSums = M.sum(1).getA()[0]
        for i in MSums:
            self.assertEqual(i,1.)
    
    def testANOVA(self):
        # Trivial case
        X = csr_matrix(np.ones((3,3)))
        yhat = ANOVARegress(X).todense()
        self.assertTrue((X == yhat).all(),msg="Trivial ANOVA")
         
        # Non-trivial case
        mu = np.random.uniform()
        alpha = np.random.uniform(size=2)
        alpha = np.concatenate((alpha,[-np.sum(alpha)]))
        beta = np.random.uniform(size = 2)
        beta = np.concatenate((beta,[-np.sum(beta)]))
        X = lil_matrix(mu * np.ones((3,3)))
        for i in xrange(3):
            for j in xrange(3):
                X[i,j] += alpha[i] + beta[j]
        yhat = ANOVARegress(csr_matrix(X)).todense()
        for i in xrange(3):
            for j in xrange(3):
                self.assertAlmostEqual(X[i,j],yhat[i,j],msg="Non trivial ANOVA")
         
if __name__ == '__main__':
    unittest.main()       
    