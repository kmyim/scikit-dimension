from .._commonfuncs import get_nn, GlobalEstimator
import numpy as np
from scipy.optimize import curve_fit
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from sklearn.utils.graph import graph_shortest_path

class AccGraphDist(GlobalEstimator):
    """"
    Estimator calculating intrinsic dimensionality by fitting distribution of geodedsic pairwise distances

    References:
    "Accurate estimation of intrinsic dimension using graph distances: unraveling the geometric complexity of datasets" 
    Scientific Report, 6, 31377 (2016)
    https://www.nature.com/articles/srep31377
    """
    def __init__(self, n_neighbors=None, eps=None, metric="euclidean", model_density_function=None, density_estimator=None):
        """
        Parameters
        ----------
        n_neighbors : int, optional, default: 5
            Number of neighbors used while constructing kNN graph.
        """
        self.n_neighbors = n_neighbors
        self.eps = eps
        self.model_density_function = model_density_function
        self.metric = metric
        self.density_estimator = density_estimator

    def fit(self, X, y=None, n_jobs=1):
        """
        Fit the estimator to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) or (n_samples, n_samples)
            The input data.

        y : Ignored
            Not used, present here for API consistency by convention.

        n_jobs : int, optional, default: 1
            The number of jobs to run in parallel for both `fit` and `predict`.
            If -1, then the number of jobs is set to the number of CPU cores.
        """
        geodesic_distances = AccGraphDist._estimate_geodesic_distances(X, self.n_neighbors)
        r_max = AccGraphDist._find_r_max(geodesic_distances) # Find maximum of pairwise distribution desnsity function
        std_distances = 2
        x_data, density_val = AccGraphDist._prepare_data_to_fit(geodesic_distances, r_max, std_distances)
        popt, _ = curve_fit(AccGraphDist._density_function, x_data, density_val)
        self.dimension_ = popt[0]
    
    def _estimate_geodesic_distances(self, X):
        """"
        Estimate geodesic distances between points using graph distances.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        
        n_neighbors : int
            Number of neighbors used while constructing kNN graph.
        
        Returns
        -------
        array-like, shape (n_samples, n_samples)
            Pairwise geodesic distances between points in the dataset.
        """
        if self.n_neighbors is None:
            dist_graph = kneighbors_graph(X, n_neighbors, mode='distance', metric=self.metric)
        if self.eps is None:
            dist_graph = radius_neighbors_graph(X, eps, mode='distance', metric=self.metric)
        geodesic_distances = graph_shortest_path(knn_graph, method='D', directed=False)
        return geodesic_distances
    
    @staticmethod
    def _find_r_max(geodedsic_distances):
        """
        Find maximum of pairwise distribution density function.

        Parameters
        ----------
        geodedsic_distances : array-like, shape (n_samples, n_samples)
            Pairwise geodesic distances between points in the dataset.

        Returns
        -------
        float
            Maximum of pairwise distribution density function.
        """

        r_max = 0
        return r_max
    
    @staticmethod
    def _hypersphere_geodesic_dist_density(r, d):
        """
        Calculate density of geodesic distances between points on a hypersphere of dimension d.

        Parameters
        ----------
        r : float
            Geodesic distance between two points on a hypersphere.

        d : int
            Dimension of the hypersphere.

        Returns
        -------
        float
            Density of geodesic distances between points on a hypersphere of dimension d.
        """
        return (np.sin(x))**(d-1)#2 * x * (np.sin(x))**(d-1) / (np.pi**d * np.math.factorial(d-1))





