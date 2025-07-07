import numpy as np
from skdim._commonfuncs import GlobalEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array
from scipy.special import betainc
from scipy.stats import hmean
from pynverse import inversefunc
from collections.abc import Iterable
from joblib import Parallel, delayed

class WODCap(GlobalEstimator):
    def __init__(self, k = 10, n_jobs = -1, aggr = 'mean', metric = 'euclidean'):

        if isinstance(k, Iterable):
            self.multiple_ks = True
            self.ks = np.array(k)
            self.maxk = max(k)
        else:
            self.maxk = k
        self.n_jobs = n_jobs        
        self.aggr = aggr
        self.metric = metric

        beta = lambda x: betainc((x+1)/2, 1/2, 3/4, out=None)

        self._invbeta = inversefunc(beta, domain = 0, open_domain = True)

    def fit(self, X, y=None):

        self._check_params()
        X = check_array(X, ensure_2d=True)

        nbrs_ = NearestNeighbors(n_neighbors=self.maxk, algorithm='auto', metric = self.metric, n_jobs = self.n_jobs).fit(X)
        
        knn_list = nbrs_.kneighbors(X, return_distance=False)
        
        least_common_sizes = np.array(Parallel(n_jobs=self.n_jobs)(
            delayed(self._find_intersection)(i, knn_list) for i in range(X.shape[0]))) # n_samples x n ks
        
        if self.multiple_ks:
            if self.aggr == 'all':
                s = self._aggregate(least_common_sizes, self.aggr) / self.ks.reshape([1,-1])  # (4, n ks) / (1, n ks)
            else:
                s = self._aggregate(least_common_sizes, self.aggr) / self.ks
        else:
            s = self._aggregate(least_common_sizes, self.aggr) / self.maxk

        if self.aggr == 'all':
            dummy = ['mean', 'median', 'hmean']
            self.dimension_ = {dummy[i]: self._invbeta(s_i) for i, s_i in enumerate(s)}
        else:
            self.dimension_ = self._invbeta(s)
        return self
    
    def _find_intersection(self, idx, neighlist):
        """Find the intersection of neighbors for a given index."""
        
        if self.multiple_ks:
            klist = []
            for k in self.ks:
                neigh_idx = set(neighlist[idx,:k])
                klist.append(min([len(neigh_idx.intersection(set(neighlist[j,:k]))) for j in neigh_idx if j != idx]))
            return klist

        else:   
            neigh_idx = set(neighlist[idx])
            return min([len(neigh_idx.intersection(set(neighlist[j]))) for j in neigh_idx if j != idx])
    
    @staticmethod
    def _aggregate(X, aggr):
        """Aggregate the results based on the specified aggrination method."""
        if aggr == 'mean':
            return np.mean(X, axis=0)
        elif aggr == 'median':
            return np.median(X, axis=0)
        elif aggr == 'hmean':
            return hmean(X, axis=0)
        elif aggr == 'all': 
            return np.array([np.mean(X, axis=0), np.median(X, axis=0), hmean(X, axis=0)])
        else:
            raise ValueError(f"Unknown combination method: {aggr}")
    
    def _check_params(self):
        """Check parameters for validity."""
        if self.multiple_ks:
            if not all(isinstance(k, int) or isinstance(k, np.int64) and k > 0 for k in self.ks):
                print(self.ks)
                raise ValueError("All elements in 'k' must be positive integers.")
        if not (isinstance(self.maxk, int) or isinstance(self.maxk, np.int64)) or self.maxk <= 0:
            raise ValueError("Parameter 'k' must be a positive integer.")
        if not isinstance(self.n_jobs, int) or self.n_jobs < -1:
            raise ValueError("Parameter 'n_jobs' must be an integer >= -1.")
        if self.aggr not in ['mean', 'median', 'hmean', 'all']:
            raise ValueError("Parameter 'aggr' must be one of ['mean', 'median', 'hmean', 'all'].")

    