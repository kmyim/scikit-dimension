import numpy as np
import random
from functools import reduce

from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_array
from scipy.sparse.csgraph import minimum_spanning_tree

from sklearn.utils.validation import check_array
from sklearn.linear_model import LinearRegression

from .._commonfuncs import GlobalEstimator


class PH(GlobalEstimator):
    """Intrinsic dimension estimation using the PHdim algorithm. 
        The final step of the algorithm involves fitting a straight line to data. 
        User should plot self.x_ against self.y_ to verify goodness of fit, and if needs be fit the straight line to a subset of x_ vs y_ to improve the inferrence. 
        
    Parameters
    ----------  
    alpha: float
        Persistence power
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
    seed: int
        random seed for subsampling

    Attributes
    ----------
    x_: 1d array 
        np.array with the log(n) values. 
    y_: 1d array 
        np.array with the log(E) values. 
    reg_: sklearn.linear_model.LinearRegression
        regression object used to fit line to log E vs log n
    """
    def __init__(self,  alpha = 1.0, n_range_min = 0.5, n_range_max = 1, range_type = 'fraction', nsteps = 100, subsamples = 10, metric = 'euclidean', random_state =12345):
        self.alpha = alpha
        self.n_range_min = n_range_min
        self.n_range_max = n_range_max
        self.range_type = range_type
        self.nsteps = nsteps
        self.subsamples = subsamples
        self.metric = metric 
        self.random_state = random_state

    def fit(self, X, y=None):
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
        self.score_: float
            Regression score
        """
        if self.range_type == 'fraction':
            self.nmin = int(np.ceil(self.n_range_min * X.shape[0]))
            self.nmax = int(np.ceil(self.n_range_max * X.shape[0]))
        elif self.range_type == 'num':
            self.nmin, self.nmax = self.n_range_min, self.n_range_max
        else:
            raise ValueError("range_type should either be 'fraction', or 'num'.")
        
        
        self.subsamplerange = np.ceil(np.linspace(self.nmin,self.nmax, self.nsteps)).astype(int)
        self._check_params(X)

        X = check_array(X, ensure_min_samples=self.subsamplerange[-1], ensure_min_features=2)

        self.dimension_ = self._phEst(X)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def _phEst(self, X):

        random.seed(self.random_state)
        
        D = squareform(pdist(X))
        D = np.triu(D)

        E = np.vstack([self._ph(D, n) for n in self.subsamplerange])
        #E = np.array(reduce(lambda xs, ys: xs + ys, E)) #flatten

        x = np.repeat(self.subsamplerange, self.subsamples).reshape([-1,1])

        self.x_ = np.log(x)
        self.y_ = np.log(E)

        if isinstance(self.alpha, list) or isinstance(self.alpha, tuple):
            dim = []
            for k in range(len(self.alpha)):
                reg = LinearRegression(fit_intercept = True).fit(self.x_, self.y_[:,k])
                dim.append(np.divide(self.alpha[k],(1-reg.coef_[0])))
        elif isinstance(self.alpha, float) or isinstance(self.alpha, int):
            reg = LinearRegression(fit_intercept = True).fit(self.x_, self.y_.reshape([-1]))
            dim = np.divide(self.alpha,(1-reg.coef_[0]))

        return dim
    
    def _ph(self, D, n):

        NUMPOINTS = D.shape[0]
        y = []
        for _ in range(self.subsamples):
            idx = random.sample(range(NUMPOINTS),n)
            D_sample = csr_array(D[idx,:][:,idx])
            T = minimum_spanning_tree(D_sample)
            y.append(np.sum(np.power(T.data.reshape([-1,1]), np.array(self.alpha).reshape([1,-1])), axis = 0))

        return np.array(y)


    def _check_params(self, X):
        if isinstance(self.alpha, list) or isinstance(self.alpha, tuple):
            if np.any(np.array(self.alpha) <= 0):
                raise ValueError("Alpha power parameter must be a strictly positive.")
        elif isinstance(self.alpha, float) or isinstance(self.alpha, int):
            if self.alpha <= 0:
                raise ValueError("Alpha power parameter must be a strictly positive.")
        if self.nmin <= 1  or not isinstance(self.nmin, int):
            raise ValueError("Min subsample population size must be an integer > 1.")
        if self.nsteps < 2 or not isinstance(self.nsteps, int):
            raise ValueError("Nsteps must be an integer >= 2.")
        if self.subsamples < 1  or not isinstance(self.subsamples, int):
            raise ValueError("Min number of subsamples must be an integer >= 1.")
        if self.nmax < self.nmin or not isinstance(self.nmax, int):
            raise ValueError("Max subsample population size must be an integer greater than than the min subsample population size.")
        if self.nmin > X.shape[0]:
            raise ValueError("Minimum subsample population size greater than number of points.")
        if self.nmax > X.shape[0]:
            raise ValueError("Maximum subsample population size greater than number of points.")
        if len(self.subsamplerange) < 2:
            raise ValueError("Subsample population range has fewer than two points, modify range of N or nstep to ensure there is a line to be fitted!")