from .._commonfuncs import FlexNbhdEstimator
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import DistanceMetric
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np

from joblib import effective_n_jobs, Parallel, delayed

class GeoMle(FlexNbhdEstimator):
    def __init__(self, k1 = 5, k2= 7, bootstrap_num = 20, alpha = 5e-3, interpolation_degree = 2,
        metric="euclidean",
        comb="mean",
        smooth=False,
        n_jobs=1,
        random_state = 12345):
        
         #neighbourhood size in building knn graph; inflated to ensure knn for mle in bootstrap samples

        super().__init__(
            pw_dim=True,
            nbhd_type="knn",
            metric=metric,
            comb=comb,
            smooth=smooth,
            n_jobs=n_jobs,
            n_neighbors= self.k2,
            sort_radial=False, 
            pt_nbhd_incl_pt=False
        )
        """
        Implementation of GeoMLE [Gomtsyan19]. 
        We modify the original version so that the bootstrapping is done independently per point on an expanded knn neighbourhood, instead of the whole dataset. 
        This avoids repeated computation of distance matrices or KNN nbhds for each bootstrap sample of the whole dataset. 
        Compared to implementation by authors, this issue fixes a bug involved with the bootstrapping procedure, that would confer a bias on the estimate

        Parameters
        ----------
        k1: int, optional
            Lower range (inclusive) of k nearest (distinct) neighbor neighborhood  on which MLE estimate of dimension is computed 
        k2: int, optional
            Upper range (inclusive) of k nearest (distinct) neighbor neighborhood  on which MLE estimate of dimension is computed 
        bootstrap_num : int, optional
            Number of bootstrap sets. The default is 20.
        alpha : float, optional
            Regularization parameter for Ridge regression. The default is 5e-3.
        interpolation_degree : int, optional
            Degree of interpolation polynomial. The default is 2.
        random_state: int, optional
            Random seed for bootstrapping 
        """
        self.alpha = alpha
        self.max_degree = interpolation_degree
        self.bootstrap_num = bootstrap_num
        self.k1 = k1
        self.k2 = k2
        self.random_state = random_state

    def _fit(self, X, nbhd_indices, radial_dists):

        # Check if the parameters are valid
        for idx, p in enumerate([self.k1, self.k2]):
            if not isinstance(p, int) or p < 3:
                raise ValueError("k" + str(idx+1) + " should be a positive integer at least 3.")
        if self.k1 >= self.k2:
            raise ValueError("k2 needs to be strictly greater than k1.")   
        if self.bootstrap_num <= 0 or not isinstance(self.bootstrap_num, int):
            raise ValueError("Number of bootstrap sets needs to be a positive integer")
        if self.max_degree <= 0 or not isinstance(self.max_degree, int):
            raise ValueError("Degree of interpolation polynomial has to be a positive integer ")

        np.random.seed(self.random_state)

        if isinstance(self.n_jobs, int) and not self.n_jobs in [0,1]:
            with Parallel(n_jobs=self.n_jobs) as parallel:
                 self.dimension_pw_ = parallel(
                    delayed(self.__local_geomle)(r)
                    for r in radial_dists)
        else:
            self.dimension_pw_ = np.array([self.__local_geomle(r) for r in radial_dists])
    
    def __local_geomle(self, radial_dists):
        """    
        Input parameters:
        radial_dists - radial distances from a single point of query, assume to be sorted
        
        Returns: 
        array of shape (len(X),) of regression dimensionality estimation for points in X averaged over bootstrap samples
        """

        mean_knn_radial_dist = None #store mean distance
        mean_mle = None #store mean mle estimate
        var_mle = None #store var in mle estimate
        
        for _ in range(self.bootstrap_num):
            ### parallelise this...
            ## row of radial_list = [d1 ( > 0), d2,...]  ##
            btstrp_radial_dists = self._bootstrap_order_preserving(radial_dists) #bootstrap resample of NN distances while keeping total order in array
            btstrp_mle = self._calc_local_mle_all_ks(btstrp_radial_dists, self.k1, self.k2) # mle estimates for nbhds of size k1,...,k2

            if mean_knn_radial_dist is None: #initialise
                mean_knn_radial_dist = btstrp_radial_dists /self.bootstrap_num # length (k2- k1+ 1)
            else:
                mean_knn_radial_dist += btstrp_radial_dists /self.bootstrap_num #update

            if mean_mle is None:
                mean_mle = btstrp_mle / self.bootstrap_num # length (k2- k1+ 1)
            else:
                mean_mle += btstrp_mle / self.bootstrap_num
            
            if var_mle is None:
                var_mle = btstrp_mle ** 2/ self.bootstrap_num
            else:
                var_mle += btstrp_mle ** 2/ self.bootstrap_num
        
        var_mle -=  mean_mle **2 #length (k2- k 1+ 1) subtract mean**2 to get variance from mean(X**2)
        return self._calc_local_estimate_from_regression(mean_mle, var_mle, mean_knn_radial_dist)

    def _calc_local_estimate_from_regression(self, mean_mle, var_mle, mean_knn_radial_dist):
        # per point! not over all
        ### TO DO NICE POLYNOMIAL BASIS FOR BETTER NUMERICS? 

        weights = np.divide(1, var_mle)
        X = np.power(mean_knn_radial_dist.reshape([-1,1]), np.arange(1, self.max_degree+ 1)) # extrapolation points x features; row = ith distance, column the power; 
        ridge_reg = Ridge(alpha=self.alpha, fit_intercept=True)
        ridge_reg.fit(X, mean_mle, weights)
        return ridge_reg.intercept_
    
    @staticmethod
    def _bootstrap_order_preserving(a):
        
        idx = np.random.choice(len(a), len(a), replace=True)
        filt = [0]*len(a)
        for i in idx:
            filt[i]+=1

        return np.repeat(a, filt)
        

    @staticmethod
    def _calc_local_mle_all_ks(dlist, k1, k2):
        #assume dlist sorted in increasing order, which is true for knn
        #(i-1)th entry in dlist is the ith nearest distinct neighbour
        #j-1 entry in cumsum = sum(log(d_i), i = 1,..., j )
        logsum = np.cumsum(np.log(dlist))[k1-1:k2] # = [sum(log(d_i, i = 1,...,k1), ..., sum(log(d_i, i = 1,...,k2)]. length #k2- k1 + 1
        lograd = np.arange(k1, k2 + 1) * np.log(dlist[k1-1:k2]) # for each k in k1,...k2, compute k*log(d_k)
        allminv = lograd - logsum
        #returns mle estimates for k1,...,k2
        return np.divide(np.arange(k1-2, k2-1), allminv) # (k1-2,..., k2-2) / allminv; use k-2 (instead k-1) to get rid of the asymptotic bias



#### TO DO ####
