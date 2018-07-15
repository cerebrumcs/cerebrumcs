'''
'''

import numpy as np
from sklearn import metrics
from sklearn.metrics import pairwise as pw 


class TWED():
    pass


class DTW():

    def __init__(self, similarity_metric = "l2"):
        self.sm = similarity_metric
        
    
    def calculate(self, X1, X2):
        '''
            calculates the dtw value and the dtw-warping path
            
            Arguments:
            ----------
            X1 : First sequence to compare
            X2 : Second sequence to compare
            
            Returns: a tuple which contains the dtw-value and a list representing the warping-path
        '''
        M, N = len(X1), len(X2)
        
        # calculates the pairwise costs between the sequence entries of X1 and X2         
        cost_matrix = np.array([[self.local_costs(x1,x2) for x1 in X1] for x2 in X2])
        
        # determine dtw values via dtw matrix
        dtw_matrix = self.calculate_dtw_matrix(cost_matrix)
        
        # determine minimal warping path
        warping_path = self.get_warping_path(dtw_matrix)
        
        dtw_value = dtw_matrix[M,N]
        
        return dtw_value, warping_path 
    
    
    def local_costs(self, x1, x2):
        if self.sm == "l1":
            return np.sum(np.abs(x1 - x2))
        elif self.sm == "l2":
            return np.sqrt(np.sum(np.square(np.abs(x1 - x2))))
        
        return -1
    
    
    def calculate_dtw_matrix(self, cost_matrix):
        '''
            dtw is defined as the costs of the warping path having the minimal costs.
            Let X1 and X2 be two sequences, and M, N the length of X1, X2, the 
            a mapping between X1, X2 is a called a valid warping path, if following conditions are satisfied:
            1. path maps (1,1) and (M,N) # ends are mapped to each other,
            2. (i+1,j+1) - (i,j) = (1,0),(0,1) or (1,1) # no symbol is bypassed, consistency of the order.    
            
            Arguments:
            ---------
                cost_matrix : matrix containing the pairwise costs.
                
            Returns: DTW-Matrix which contains the minimal cost to step in a mapping (x1,x2)

        '''        
        
        M, N = cost_matrix.shape
        
        dtw_matrix = np.zeros(shape = (M,N))
        for n in range(N):
            for m in range(M):
                cost = cost_matrix[m,n]
                
                costs_neighbors = []
                if m > 0:
                    costs_neighbors += [dtw_matrix[m-1,n]]
                if n > 0:
                    costs_neighbors += [dtw_matrix[m,n-1]]
                if m > 0 and n > 0:
                    costs_neighbors += [dtw_matrix[m-1,n-1]]
                
                min_cost_neighbor = np.min(costs_neighbors) if len(costs_neighbors) > 0 else 0
                
                dtw_matrix[m,n] = cost + min_cost_neighbor 
        
        return dtw_matrix

    
    def get_warping_path(self, dtw_matrix):
        '''
            determines the minimal warping path by a give dtw matrix
            
            Arguments:
            ----------
            dtw_matrix : the dtw matrix, generated using calculate_dtw_matrix() method
            
            Returns: a list of tupels representing the indices of the best mapping between the sequences.
                The list will always starts with (1,1) and will always ends with (M,N). 
            
        '''
        M, N = dtw_matrix.shape        
        warping_path = [(M,N)]

        while True:
            
            if M > 0 and N > 0:
                dtw_sub_matrix = dtw_matrix[M-1:M,N-1:N]
                m, n = np.argwhere(dtw_sub_matrix == np.min(dtw_sub_matrix))[0]
            elif M > 0:
                m, n = 1,0
            elif N > 0:
                m, n = 0,1
            else:
                break
                        
            M, N = M - m, N - n
            warping_path += [(M,N)]
            
        return list(reversed(warping_path))