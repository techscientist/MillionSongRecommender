import numpy as np
import scipy.io
from scipy.sparse import lil_matrix, csr_matrix

def genMSDData(basePath = 'C:/Users/William/Downloads/'):
    """ Reads in data from kaggle, outputs n by p dimensional array A, where n is the number of users, p the number of artists, and A_ij is the proportion of 
    times user i listened to artist j. This proportion acts as my 'rating', which I try to predict later on.
    """
    userData = np.loadtxt(basePath + 'kaggle_visible_evaluation_triplets/kaggle_visible_evaluation_triplets.txt',
                          dtype=np.str)
    uniqueUsers = {a:i for i,a in enumerate(np.unique(userData[:,0]))}
    uniqueTracks = {a:i for i,a in enumerate(np.unique(userData[:,1]))}
    
    result = lil_matrix((len(uniqueUsers),len(uniqueTracks)))
    for user, track, value in userData:
        result[uniqueUsers[user],uniqueTracks[track]] += float(value)
    
    csrResult = csr_matrix(result)
    rowSums = np.array([1 / np.sum(csrResult[i,:].todense()) for i in xrange(csrResult.shape[0])])
    
    diags = lil_matrix((len(uniqueUsers),len(uniqueUsers)))
    diags.setdiag(rowSums)
    
    csrResult = diags * csrResult
    
    scipy.io.mmwrite(basePath + 'MSDMatrix.mtx', csrResult)

def readMSDData(basePath = 'C:/Users/William/Downloads/'):
    return(csr_matrix(scipy.io.mmread(basePath + 'MSDMatrix.mtx')))

        