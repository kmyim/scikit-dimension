import numpy as np
from .._commonfuncs import FlexNbhdEstimator
from sklearn.utils.parallel import Parallel, delayed
from joblib import effective_n_jobs
from ..id import lPCA


class LinZha(FlexNbhdEstimator):
    """
    With either knn or epsilon neighbourhood

    Tong Lin and Hongbin Zha.
    Riemannian manifold learning.
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 30(5):796–809, 5 2008.
    https://doi.org/10.1109/TPAMI.2007.70735
    """

    def _fit(
        self,
        X,
        nbhd_indices,
        nbhd_type,
        metric,
        radial_dists,
        radius=1.0,
        n_neighbors=5,
        n_jobs=None,
    ):
        """Pointwise dimension should be largest simplex with an element here
        aggregation method should be max."""
        if nbhd_type not in ["eps", "knn"]:
            raise ValueError("Neighbourhood type should either be knn or eps.")

        if effective_n_jobs(n_jobs) > 1:
            with Parallel(n_jobs=n_jobs) as parallel:
                # Asynchronously apply the `fit` function to each data point and collect the results
                results = parallel(
                    delayed(self._local_lin_zha_dim)(X, idx, nbhd)
                    for idx, nbhd in enumerate(nbhd_indices)
                )
            self.dimension_pw_ = np.array(results)
        else:
            self.dimension_pw_ = np.array(
                [
                    self._local_lin_zha_dim(X, idx, nbhd)
                    for idx, nbhd in enumerate(nbhd_indices)
                ]
            )

    @staticmethod
    def _visible_neighbors(X, idx, nbhd, radial_dist):
        if len(nbhd) == 0:
            return 0
        neighbors = np.take(X, nbhd, 0)
        vectors = neighbors - X[idx]

        visibility=[]
        for i, vector in enumerate(vectors):
            for j, other_vector in enumerate(vectors):
                if j==i:
                    continue
                onward_vector = vector - other_vector
                if np.dot(onward_vector, other_vector) >= 0:
                    visibility.append(False)
                    break
            visibility.append(True)
        visibility = np.array(visibility)

        visible_nbhd = nbhd[visibility]
        radial_dist = radial_dist[visibility]

        return visible_nbhd, radial_dist

    @staticmethod
    def _safe_neighbors(X, idx, nbhd, radial_dist):
        nbhd, radial_dist = LinZha._visible_neighbors(X, idx, nbhd, radial_dist)
        nbhd = np.concatenate(np.array([idx]), nbhd)
        radial_dist = np.concatenate(np.array([0.0]), radial_dist)
        neighbors = np.take(X, nbhd, axis=0)
        #TODO Fix this sorting issue
        sorted_indices = radial_dist.argsort()
        sorted_neighbors = neighbors[sorted_indices]
        sorted_distances = radial_dist[sorted_indices]
        incremental_dims = []
        for i, _ in enumerate(sorted_neighbors):
            data = sorted_neighbors[0:i,:]
            incremental_dims.append(lPCA(ver="FO", alphaFO=0.05).fit_transform(data))
        # get vector of jumps
        # compare to average jumps



