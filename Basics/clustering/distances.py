'''
'''

import numpy as np

def eucl_dist(x1,x2):
    return np.sqrt(np.sum(np.square(x1 - x2)))


def mahalanobis_dist(x1, x2, cov):
    pass  
    