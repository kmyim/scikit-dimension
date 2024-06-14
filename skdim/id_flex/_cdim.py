import inspect
import numpy as np
import scipy
from .._commonfuncs import FlexNbhdEstimator
from sklearn.metrics import DistanceMetric
from ..errors import EstimatorFailure

class CDim(FlexNbhdEstimator):
    '''
    Conical dimension method with either knn or epsilon neighbourhood

    Yang et al.
    Conical dimension as an intrinsic dimension estimator and its applications
    https://doi.org/10.1137/1.9781611972771.16
    '''


    def _fit(self, X, nbhd_indices, nbhd_type, metric, radial_dists, radius = 1.0, n_neighbors = 5,):

        if nbhd_type not in ['eps', 'knn']: raise ValueError('Neighbourhood type should either be knn or eps.')

        self.dimension_pw_ = np.array([self._mle_formula(dlist, nbhd_type, radius) for dlist in radial_dists])


    @staticmethod
    def _get_nbhd_vectors(X, nbhd_indices):
        X[nbhd_indices]

    @staticmethod
    def _mle_formula(dlist, nbhd_type, radius):

        N = len(dlist)
        
        if N > 0:
            if nbhd_type != 'eps': # defaults to knn
                radius = np.max(dlist)
            
            minv = N*np.log(radius)-np.sum(np.log(dlist))

            if minv < 1e-9:
                raise EstimatorFailure("MLE estimation diverges.")
            else:
                if nbhd_type == 'eps':
                    return np.divide(N, minv)
                else: #knn
                    return np.divide(N-1,minv)
        else:
            return np.nan
                


        
    
