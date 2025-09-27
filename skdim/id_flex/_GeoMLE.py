from .._commonfuncs import FlexNbhdEstimator
from sklearn.linear_model import Ridge
import numpy as np

from joblib import Parallel, delayed

class GeoMle(FlexNbhdEstimator):
    def __init__(self, k1 = 5, k_steps = 10, bootstrap_nbhd = None, bootstrap_num = 20, alpha = 5e-3, interpolation_degree = 2, weight_reg= 1e-3,
        metric="euclidean",
        comb="mean",
        smooth=False,
        n_jobs=1,
        random_state = 12345):
        
        
        """
        Implementation of GeoMLE [Gomtsyan19]. 
        GeoMLE is a local MLE based estimator that uses bootstrapping and polynomial interpolation to reduce variance and bias of the basic Levina-Bickel MLE.
        GeoMLE aggregates dimension estimates over a range of k-nearest neighbour sizes (k1,...,k2) using a polynomial regression fitted on bootstrap samples of the MLE estimates.
        We modify the original version so that the bootstrapping is done independently per point on an expanded knn neighbourhood, instead of the whole dataset. 
        This avoids repeated computation of distance matrices or KNN nbhds for each bootstrap sample of the whole dataset. 
        Compared to implementation by authors, we implement true bootstrapping (described in the paper) as opposed to sub-sampling without replacement.

        Parameters
        ----------
        k1: int, optional
            Lower range (inclusive) of k nearest (distinct) neighbor neighborhood  on which MLE estimate of dimension is computed 
        k_steps: int, optional
            k1 + k_steps is the upper range (inclusive) of k nearest (distinct) neighbor neighborhood, on which MLE estimate of dimension is computed 
        bootstrap_nbhd : int, optional
            Size of neighbourhood (in terms of number of nearest neighbours) used for bootstrapping. If None, set to k2 + 5. The default is None.
        bootstrap_num : int, optional
            Number of bootstrap sets. The default is 20. If bootstrap_num=0, then equal weights are applied to all points in the subsequent regression step of the estimator.
        alpha : float, optional
            Regularization parameter for Ridge regression. The default is 5e-3.
        interpolation_degree : int, optional
            Degree of interpolation polynomial. The default is 2.
        weight_reg: float, optional
            weights on points in ridge regression are given by 1/max(standard deviation in bootstrap,  weight_reg). The default is weight_reg = 1e-3.
        metric : str, optional
            Metric to use for distance computation. The default is "euclidean".
        comb : str, optional
            Method to combine local dimension estimates. The default is "mean".
        smooth : bool, optional
            Whether to apply smoothing to the local dimension estimates. The default is False.
        n_jobs : int, optional
            Number of parallel jobs to run. The default is 1.
        random_state: int, optional
            Random seed for bootstrapping 
        """
        self.alpha = alpha
        self.max_degree = interpolation_degree
        self.bootstrap_num = bootstrap_num
        
        self.k1 = k1
        self.k2 = k1 + k_steps
        
        if bootstrap_nbhd is None:
            self.bootstrap_nbhd = self.k2 + 5 #default neighbourhood for bootstrapping
        else: 
            self.bootstrap_nbhd = bootstrap_nbhd
            
        self.random_state = random_state
        self.weight_reg =weight_reg

        super().__init__(
            pw_dim=True,
            nbhd_type="knn",
            metric=metric,
            comb=comb,
            smooth=smooth,
            n_jobs=n_jobs,
            n_neighbors= self.bootstrap_nbhd,
            sort_radial=False, 
            pt_nbhd_incl_pt=False
        )

    def _fit(self, X, nbhd_indices, radial_dists):

        # Check if the parameters are valid
        if not isinstance(self.k1, int) or  self.k1 >= X.shape[0]-1 or self.k1 < 3:
            raise ValueError("k1 should be a positive integer at least 3 and at most (number of points -2).")
        if self.k1 >= self.k2 or not isinstance(self.k2, int) or  self.k2 >= X.shape[0] or self.k2 < 3:
            raise ValueError("k2 needs to be  needs to be a positive integer at least 3 and at most (number of points - 1).")   
        if self.bootstrap_nbhd < self.k2 or not isinstance(self.bootstrap_nbhd, int) or  self.bootstrap_nbhd < 3:
            raise ValueError("Bootstrap neighbourhood must be at least k2.")  
        if self.bootstrap_num < 0 or not isinstance(self.bootstrap_num, int):
            raise ValueError("Number of bootstrap sets needs to be a non-negative integer.")
        if self.max_degree <= 0 or not isinstance(self.max_degree, int):
            raise ValueError("Degree of interpolation polynomial has to be a positive integer.")
        if self.max_degree >= self.k2 - self.k1 + 1:
            raise ValueError("Degree of interpolation polynomial must be strictly less than (k_steps + 1).")
        if self.alpha < 0:
            raise ValueError("Regularization parameter alpha must be non-negative.")
        if self.weight_reg <= 0:
            raise ValueError("weight_reg must be positive.")

        np.random.seed(self.random_state)

        if isinstance(self.n_jobs, int) and not self.n_jobs in [0,1]:
            with Parallel(n_jobs=self.n_jobs) as parallel:
                    res = parallel(
                        delayed(self.__local_geomle)(r)
                        for r in radial_dists)
        else:
            res = [self._calc_local_mle_all_ks(r, self.k1, self.k2) for r in radial_dists]
        
        self.dimension_pw_ = np.array(res)



    
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
        if self.bootstrap_num > 0:
            for _ in range(self.bootstrap_num):
                ## row of radial_list = [d1 ( > 0), d2,...]  ##
                btstrp_radial_dists = self._bootstrap_order_preserving(radial_dists) #bootstrap resample of NN distances while keeping total order in array
                btstrp_mle = self._calc_local_mle_all_ks(btstrp_radial_dists, self.k1, self.k2) # mle estimates for nbhds of size k1,...,k2

                if mean_knn_radial_dist is None: #initialise
                    mean_knn_radial_dist = btstrp_radial_dists /self.bootstrap_num # length k2
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
        else:
            mean_knn_radial_dist = radial_dists # length k2
            mean_mle = self._calc_local_mle_all_ks(radial_dists, self.k1, self.k2) # length (k2- k1+ 1)
            var_mle = np.ones_like(mean_mle) # no bootstrapping, so set variance to 1 to give equal weights in regression
        
        
        return self._calc_local_estimate_from_regression(mean_mle, var_mle, mean_knn_radial_dist[self.k1-1:self.k2])

    def _calc_local_estimate_from_regression(self, mean_mle, var_mle, mean_knn_radial_dist):
        # per point! not over all
        weights = np.divide(1, np.maximum(np.sqrt(var_mle),self.weight_reg))
        X = np.vander(mean_knn_radial_dist, self.max_degree + 1, increasing = True)[:,1:] # extrapolation points x features; row = ith distance, column the power;
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
        # compute mle estimates for k = k1,...,k2
        # assume dlist sorted in increasing order, which is true for knn
        #(i-1)th entry in dlist is the ith nearest distinct neighbour
        #j-1 entry in cumsum = sum(log(d_i), i = 1,..., j )
        logsum = np.cumsum(np.log(dlist))[k1-2:k2-1] # = [sum(log(d_i, i = 1,...,k1-1), ..., sum(log(d_i, i = 1,...,k2-1)]. length #k2- k1 + 1
        lograd = np.arange(k1-1, k2) * np.log(dlist[k1-1:k2]) # for each k in k1,...k2, compute (k-1)*log(d_k)
        allminv = lograd - logsum
        if np.any(allminv < 0):
            raise ValueError("MLE estimate is ill-define due to non-distinct nearest neighbours. Try increasing k1 or bootstrap_nbhd parameters.")
        #returns mle estimates for k1,...,k2
        return np.divide(np.arange(k1-2, k2-1), allminv) # (k1-2,..., k2-2) / allminv; use k-2 (instead k-1) to get rid of the asymptotic bias



