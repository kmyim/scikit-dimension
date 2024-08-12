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

    Parameters
    ----------  
    nmin: int
        Minimum subsample size
    nstep: int
        Difference between successive subsample sizes 
    alpha: float
        Persistence power
    k: int
        Number of random subsamples per size
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
    def __init__(self, nmin = 2, nstep = 1, alpha = 1.0, k = 10, metric = 'euclidean', seed =12345):
        self.alpha = alpha
        self.nmin = nmin
        self.nstep = nstep
        self.k = k
        self.metric = metric 
        self.seed = seed

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
        self.score_: float
            Regression score
        """
        X = check_array(X, ensure_min_samples=self.nmin + self.nstep + 1, ensure_min_features=2)

        self.dimension_, self.reg_ = self._phEst(X)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def _phEst(self, X):

        NUMPOINTS = X.shape[0]
        random.seed(self.seed)
        
        D = squareform(pdist(X))
        D = np.triu(D)

        nrange = range(self.nmin, NUMPOINTS, self.nstep)

        E = [self._ph(D, n) for n in nrange]
        E = np.array(reduce(lambda xs, ys: xs + ys, E)) #flatten

        x = np.repeat(nrange, self.k).reshape([-1,1])

        self.x_ = np.log(x)
        self.y_ = np.log(E)

        reg = LinearRegression(fit_intercept = True).fit(self.x_, self.y_)
        dim = np.divide(self.alpha,(1-reg.coef_))[0]
        return dim, reg
    
    def _ph(self, D, n):

        NUMPOINTS = D.shape[0]
        y = []
        for _ in range(self.k):
            idx = random.sample(range(NUMPOINTS),n)
            D_sample = csr_array(D[idx,:][:,idx])
            T = minimum_spanning_tree(D_sample)
            y.append(np.sum(np.power(T.data, self.alpha)))

        return y


