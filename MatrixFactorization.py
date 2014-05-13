import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix

def ANOVARegress(M):
    """ 
    Perform simple ANOVA regression on 2 treatments without interaction effects
    Input:
     - M is a sparse matrix, each column denotes a treatment from group A, and each row a treatment from group B
    Output:
     a sparse matrix of predicted values for all entries where data are given
    """
    nonZeroM = M != 0
    mu = M.sum() / nonZeroM.sum()
    
    beta = (M.sum(0) / nonZeroM.sum(0) - mu).getA1()
    alpha = (M.sum(1) / nonZeroM.sum(1) - mu).getA1()
    
    result = lil_matrix(M.shape)
    nonZero = M.nonzero()
    for i,j in zip(nonZero[0],nonZero[1]):
        result[i,j] = mu + beta[j] + alpha[i]  
        
    return(result)

def MatrixFactorization(R,K,eps = 0.005):
    """
    Compute a standard matrix factorization R = PQ^T, with P NxK, Q MxK, K is the inputed number of features
    Alternating gradient descent is used to solve the minimization problem
    Input:
    - R is a sparse matrix, 
    - K is the number of columns in the created matrices P and Q
    - eps is the minimum percent improvement in the loss function to avoid terminating the optimization
    Output:
    P, Q matrices described above, and a sparse matrix containing fitted training values
    """
    # Standardize input 
    R = R - ANOVARegress(R)
    
    #oldLoss = 
    #while (oldLoss - newLoss) / oldLoss < eps
    # Perform 
