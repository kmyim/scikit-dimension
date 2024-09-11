import numpy as np
import skdim
import scipy
from skdim._commonfuncs import GlobalEstimator
from sklearn.neighbors import NearestNeighbors
from scipy import special
from pynverse import inversefunc


class WODCap(GlobalEstimator):
    def __init__(self, k):
        self.k = k
        

    def fit(self, X, y=None):
        self.nbrs_ = NearestNeighbors(n_neighbors=self.k+1, algorithm='auto').fit(X)
        
        distances, indices = self.nbrs_.kneighbors(X)
        
        n_points = X.shape[0]
        self.knn_matrix_ = np.zeros((n_points, n_points), dtype=int)
        
        for i in range(n_points):
            self.knn_matrix_[i, indices[i]] = 1
        
        self.knn_list_ = []
        for i in range(self.knn_matrix_.shape[0]):
            neighbors = np.nonzero(self.knn_matrix_[i])[0]
            self.knn_list_.append(neighbors.tolist())

        self.least_common_neighbors_ = []
        self.least_common_sizes_ = []
        for i, neighbors in enumerate(self.knn_list_):
            min_common_elements = float('inf')
            least_common_j = None
            least_common_size = None
            
            for j in neighbors:
                if j == i:  
                    continue
                j_neighbors = self.knn_list_[j]
                common_elements = len(set(neighbors).intersection(set(j_neighbors)))
                
                if common_elements < min_common_elements:
                    min_common_elements = common_elements
                    least_common_j = j
                    least_common_size = common_elements
            
            self.least_common_neighbors_.append(least_common_j)
            self.least_common_sizes_.append(least_common_size)

        self.sumk_ = sum(self.least_common_sizes_) / ((self.k+1) * len(self.least_common_sizes_))
        return self
    
    def fit_transform(self, X):
        beta = lambda x: scipy.special.betainc((x+1)/2, 1/2, 3/4, out=None)
        
        invbeta = inversefunc(beta)
        
        dimension = invbeta(self.sumk_)
        return dimension
