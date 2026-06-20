#
# BSD 3-Clause License
#
# Copyright (c) 2020, Jonathan Bac
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.utils.validation import check_array
from .._commonfuncs import GlobalEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression


class KNN(GlobalEstimator):
    # SPDX-License-Identifier: MIT, 2017 Kerstin Johnsson [IDJohnsson]_
    """Intrinsic dimension estimation using the kNN algorithm. [Carter2010]_ [IDJohnsson]_

    This is a simplified version of the kNN dimension estimation method described by Carter et al. (2010), 
    the difference being that block bootstrapping is not used.

    Parameters
    ----------
    X: 2D numeric array
        A 2D data set with each row describing a data point.
    k: int
        Number of distances to neighbors used at a time.
    ps: 1D numeric array
        Vector with sample sizes; each sample size has to be larger than k and smaller than nrow(data).
    M: int, default=1
        Number of bootstrap samples for each sample size.
    gamma: int, default=2
        Weighting constant.
    """

    def __init__(self, k=None, ps=None, M=1, gamma=2):
        self.k = k
        self.ps = ps
        self.M = M
        self.gamma = gamma

    def fit(self, X, y=None):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The training input samples.
        y : dummy parameter to respect the sklearn API

        Returns
        -------
        self: object
            Returns self.
        self.dimension_: float
            The estimated intrinsic dimension
        self.residual_: float
            Residuals
        """

        self._k = 2 if self.k is None else self.k
        self._ps = np.arange(self._k + 1, self._k + 5) if self.ps is None else self.ps

        X = check_array(X, ensure_min_samples=self._k + 1, ensure_min_features=2)

        self.dimension_, self.residual_ = self._knnDimEst(X)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def _knnDimEst(self, X):
        n = len(X)
        Q = len(self._ps)

        if min(self._ps) <= self._k or max(self._ps) > n:
            raise ValueError("ps must satisfy k<ps<len(X)")
        # Compute the distance between any two points in the X set
        dist = squareform(pdist(X))

        # Compute weighted graph length for each sample
        L = np.zeros((Q, self.M))

        for i in range(Q):
            for j in range(self.M):
                samp_ind = np.random.randint(0, n, self._ps[i])
                for l in samp_ind:
                    L[i, j] += np.sum(
                        np.sort(dist[l, samp_ind])[1 : (self._k + 1)] ** self.gamma
                    )
                    # Add the weighted sum of the distances to the k nearest neighbors.
                    # We should not include the sample itself, to which the distance is
                    # zero.

        # Least squares solution for m
        d = X.shape[1]
        epsilon = np.repeat(np.nan, d)
        for m0, m in enumerate(np.arange(1, d + 1)):
            alpha = (m - self.gamma) / m
            ps_alpha = self._ps ** alpha
            hat_c = np.sum(ps_alpha * np.sum(L, axis=1)) / (
                np.sum(ps_alpha ** 2) * self.M
            )
            epsilon[m0] = np.sum(
                (L - np.tile((hat_c * ps_alpha)[:, None], self.M)) ** 2
            )
            # matrix(vec, nrow = length(vec), ncol = b) is a matrix with b
            # identical columns equal to vec
            # sum(matr) is the sum of all elements in the matrix matr

        de = np.argmin(epsilon) + 1  # Missing values are discarded
        return de, epsilon[de - 1]


class KNNfast(GlobalEstimator):
    """Intrinsic dimension estimation using the kNN algorithm. [Carter2010]_ [IDJohnsson]_

    This is a version of the kNN dimension estimation method described by Carter et al. (2010), with block bootstrapping.
    This is designed to be comparable to the PH_knn estimator. Returns knn estimates for all k's from 1 to a given max k value.
    Uses scikit-learn NearestNeighbor and LinearRegression in computations unlike KNN.

    Parameters
    ----------
    max_k: int 
        Maximum neighbourhood size for computing knn estimates for k from 1 to max_k
    n_range: 2-tuple
        Min and Max sizes of subsamples. If range_type = 'fraction', then n_range is the min and max fractions of the number of points; if 'num', then n_range is the min and max number
    range_type: str
        Specifies whether n_range describes fraction
    nsteps: int
        number of regression subsample sizes
    subsamples: int
        Number of random subsamples per size of subsample
    metric: str
        scipy.spatial.distance metric parameter
    random_state: int
        random seed for bootstrap subsampling
    n_jobs: int 
        number of parallel processes for knn computation
    
    Attributes
    ----------
    x_: 1d array 
        np.array with the log(n) values. 
    y_: max_k x n_steps array 
        np.array with the log(<L_k>) values as rows for each k. 
    reg_: list of sklearn.linear_model.LinearRegression
        list of regression object used to fit line to log L_k vs log n
    """

    def __init__(self,  max_k = 1, n_range_min = 0.75, n_range_max = 1, range_type = 'fraction', nsteps = 10, subsamples = 10, metric = 'euclidean', random_state =12345,  n_jobs = 1):
        self.max_k = max_k
        self.n_range_min = n_range_min
        self.n_range_max = n_range_max
        self.range_type = range_type
        self.nsteps = nsteps
        self.subsamples = subsamples
        self.metric = metric 
        self.random_state = random_state
        self.n_jobs = n_jobs
    
    def fit(self, X, y = None):
        """
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The training input samples.
        y : dummy parameter to respect the sklearn API

        Returns
        -------
        self: object
            Returns self.
        self.dimension_: float
            The estimated intrinsic dimension
        self.reg_: object
            sklearn LinearRegression object
        """
        if self.range_type == 'fraction':
            self.nmin = int(np.ceil(self.n_range_min * X.shape[0]))
            self.nmax = int(np.ceil(self.n_range_max * X.shape[0]))
        elif self.range_type == 'num':
            self.nmin, self.nmax = self.n_range_min, self.n_range_max
        else:
            raise ValueError("range_type should either be 'fraction', or 'num'.")
        
        
        self._subsamplerange = np.ceil(np.linspace(self.nmin,self.nmax, self.nsteps)).astype(int)
        self._check_params(X)

        X = check_array(X, ensure_min_samples=self._subsamplerange[-1], ensure_min_features=2)

        self.dimension_ = self._knnEst(X)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self


    def _knnEst(self, X):
        '''
        Extracts dimension estimate for k = 1,..., max_k
        '''
        np.random.seed(self.random_state)
        N = X.shape[0]
        length_scaling = []
        for n_samples in self._subsamplerange:
            length_vector = np.zeros(self.max_k)
            for i in range(self.subsamples): 
                subsample_indices = np.random.choice(X.shape[0], size=n_samples , replace=False)
                X_subsampled = X[subsample_indices]
                length_vector += self._knn_length(X_subsampled, self.max_k, self.n_jobs, self.metric) #room for optimisation here: knn graph recomputed for each subsample, can we compute a big knn graph for large k and compute knn graph approximation on big knn graphs?
            length_vector /= self.subsamples
            length_scaling.append(np.log(length_vector))
        self.y_ = np.array(length_scaling).T

        id = []
        self.x_ = np.log(self._subsamplerange).reshape(-1,1)
        self.reg_ =[]
        for i in range(self.max_k):
            est = LinearRegression().fit(self.x_, self.y_[i].reshape(-1,1))
            self.reg_.append(est)
            id.append(1/(1-est.coef_[0]))
        return id
    
    def _check_params(self, X):

        if isinstance(self.max_k, int):
            if self.max_k <= 0:
                raise ValueError("kNN parameter must be a strictly positive integer.")
        else:
            raise ValueError("kNN parameter must be a strictly positive integer.")
        
        if self.range_type == 'num':
            if self.n_range_min <= 1  or not isinstance(self.nmin, int):
                raise ValueError("Min subsample population size must be an integer > 1.")
            if self.n_range_max < self.n_range_min or not isinstance(self.nmax, int):
                raise ValueError("Max subsample population size must be an integer greater than than the min subsample population size.")
        elif self.range_type =='fraction':
            if self.n_range_max < self.n_range_min :
                raise ValueError("Max subsample population fraction must be in (0,1] greater than than the min subsample fraction.")
            if (self.n_range_max > 1) or (self.n_range_max <= 0):
                raise ValueError("Max subsample population fraction must be in (0,1] greater than than the min subsample fraction.")
            if (self.n_range_min > 1) or (self.n_range_min <= 0):
                raise ValueError("Max subsample population fraction must be in (0,1] greater than than the min subsample fraction.")
        if self.nsteps < 2 or not isinstance(self.nsteps, int):
            raise ValueError("Nsteps must be an integer >= 2.")
        if self.subsamples < 1  or not isinstance(self.subsamples, int):
            raise ValueError("Min number of subsamples must be an integer >= 1.")
        
        if self.nmin > X.shape[0]:
            raise ValueError("Minimum subsample population size greater than number of points.")
        if self.nmin <= 1:
            raise ValueError("Minimum subsample population should be greater than one.")
        if self.nmax > X.shape[0]:
            raise ValueError("Maximum subsample population size greater than number of points.")
        if self.nmax <= 1:
            raise ValueError("Maximum subsample population should be greater than one.")
        if self.nmin >= self.nmax:
            raise ValueError("Maximum subsample population should be greater than minimum subsample population.")
        if len(self._subsamplerange) < 2:
            raise ValueError("Subsample population range has fewer than two points, modify range of N or nstep to ensure there is a line to be fitted!")
        if not isinstance(self.n_jobs, int):
            raise ValueError("n_jobs must be integer.")
        if not isinstance(self.random_state, int):
            raise ValueError("random state must be integer.")
       
    
    @staticmethod
    def _knn_length(X, k = 30, n_jobs = -1, metric = 'euclidean'):
        kdist, kidx = NearestNeighbors(n_neighbors = k, n_jobs= n_jobs, metric = metric).fit(X).kneighbors(X)
        kdist= kdist[:,1:]
        kidx = kidx[:,1:]
        rank_dict = dict()
        dist_dict = dict()
        for i in range(len(kdist)):
            for j in range(len(kdist[i])):
                (a,b) = tuple(sorted((i,kidx[i,j])))
                if (a,b) in rank_dict:
                    rank_dict[(a,b)] = min(rank_dict[(a,b)], j)
                else:
                    rank_dict[(a,b)] = j
                    dist_dict[(a,b)] = kdist[i,j]
        lengths = [sum(dist_dict[x] for x in dist_dict if rank_dict[x] == r) for r in range(k)]

        return np.cumsum(lengths)