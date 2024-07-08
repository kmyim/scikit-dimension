from .._commonfuncs import get_nn, GlobalEstimator
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import gamma
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
    def __init__(self, n_neighbors=None, eps=None, metric="euclidean", model_density_function=None,
        density_estimator=None):
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
        if model_density_function is None:
            self.model_density_function = AccGraphDist._scaled_hypersphere_geodesic_dist_density
        else:
            self.model_density_function = model_density_function
        if density_estimator is None:
            self.density_estimator = AccGraphDist.HistogramDensityEstimator(bins=100)
        else:
            self.density_estimator = density_estimator
    
    class HistogramDensityEstimator:
        """
        Density estimator based on histogram. (For compatibility with original code)
        The estimator fits histogram to the data and calculates density at given points.
        """
        def __init__(self, bins=10, range=None, weights=None):
            self.bins = bins
            self.range = range
            self.weights = weights

        def fit(self, X):
            self.density, self.bins = np.histogram(X, bins=self.bins, density=True, range=self.range, weights=self.weights)
        
        def max(self):
            return np.max(self.density)
        
        def score_samples(self, X):
            return np.interp(X, self.bins[:-1], self.density)

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
        # Estimate geodesic distances using graph distances
        geodesic_distances = AccGraphDist._estimate_geodesic_distances(X, self.n_neighbors)
        # Fit density estimator to the geodesic distances
        self.density_estimator.fit(geodesic_distances)
        # Calculate basic statistics of the geodesic distances
        max_distance = np.max(geodesic_distances)
        std_distances = np.std(geodesic_distances)
        # Find at which distance the density function reaches its maximum
        r_max = self._find_r_max(density_estimator, max_distance)# Find maximum of pairwise distribution density function
        # Sample uniformly interval [r_max - 2s, r_max] and calculate density at samples
        x_data, density_val = self._prepare_data_to_fit(r_max, std_distances)
        # Fit estimated density function to model density function depending on ID
        popt, _ = curve_fit(AccGraphDist._density_function, x_data, density_val)
        self.dimension_ = popt[0]
    
    def _prepare_data_to_fit(self, r_max, std_distances):
        """
        Prepare data to fit the density function.

        Parameters
        ----------
        r_max : float
            Maximum of pairwise distribution density function.

        std_distances : float
            Standard deviation of pairwise geodesic distances.

        Returns
        -------
        array-like, shape (n_samples, n_samples)
            Pairwise geodesic distances between points in the dataset.
        """
        x_data = np.linspace(r_max - 2 * std_distances, r_max, 100)
        density_val = self.density_estimator.score_samples(x_data)
        return x_data, density_val
    
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
    def _find_r_max(self, max_distance):
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

        x_grid = np.linspace(0, max_distance, 100)
        density_val = self.density_estimator.score_samples(x_grid)
        return x_grid[np.argmax(density_val)]
    


    @staticmethod
    def _scaled_hypersphere_geodesic_dist_density(r, d, r_max):
        return AccGraphDist._density_function(np.pi * r / r_max / 2 , d)
    
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
        return (np.sin(x))**(d-1) * (scipy.special.gamma( (d+1)/2)) / scipy.special.gamma(d / 2) / np.pi





