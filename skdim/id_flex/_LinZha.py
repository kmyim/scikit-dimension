import numpy as np
import networkx as nx
from .._commonfuncs import FlexNbhdEstimator
from sklearn.utils.parallel import Parallel, delayed
from joblib import effective_n_jobs
from ..id import lPCA


def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens), len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[: len(l), idx] = l
    return arr.mean(axis=-1)


class LinZha(FlexNbhdEstimator):
    """
    With either knn or epsilon neighbourhood

    Tong Lin and Hongbin Zha.
    Riemannian manifold learning.
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 30(5):796–809, 5 2008.
    https://doi.org/10.1109/TPAMI.2007.70735
    """

    def __init__(
        self,
        rho=1.5,
        nbhd_type="knn",
        metric="euclidean",
        n_jobs=1,
        radius=1.0,
        n_neighbors=5,
    ):
        """
        rho: Thresholding value: neighbors which are separated by more than rho times the mean distance between neighbors of that rank are excluded in calculating dimension
        nbhd_type: either 'knn' (k nearest neighbour) or 'eps' (eps nearest neighbour) or 'custom'
        metric: defaults to standard euclidean metric; if X a distance matrix, then set to 'precomputed'
        n_jobs: number of parallel processes in inferring local neighbourhood
        kwargs: keyword arguments, such as 'n_neighbors', or 'radius' for sklearn NearestNeighbor to infer local neighbourhoods
        """

        super().__init__(
            pw_dim=False,
            nbhd_type=nbhd_type,
            pt_nbhd_incl_pt=True,
            metric=metric,
            comb=None,
            smooth=None,
            n_jobs=n_jobs,
            radius=radius,
            n_neighbors=n_neighbors,
            sort_radial=True,
        )
        self.rho = rho

    def _fit(self, X, nbhd_indices, radial_dists):
        """Pointwise dimension should be largest simplex with an element here
        aggregation method should be max."""
        if effective_n_jobs(self.n_jobs) > 1:
            with Parallel(n_jobs=self.n_jobs) as parallel:
                # Asynchronously apply the `fit` function to each data point and collect the results
                results = parallel(
                    delayed(self._calculate_jumps)(X, idx, nbhd, radial_dists[idx])
                    for idx, nbhd in enumerate(nbhd_indices)
                )
        else:
            results = [
                self._calculate_jumps(X, idx, nbhd, radial_dists[idx])
                for idx, nbhd in enumerate(nbhd_indices)
            ]
        visible_nbhd, radial_dists, jumps = zip(*results)
        threshold_jump = self.rho * tolerant_mean(jumps)
        threshold_jump[0] = -1
        safe_nbhds = []
        for idx, result in enumerate(results):
            safe_nbhds.append(result[0][result[2] < threshold_jump])

        nbhd_graph = nx.Graph()
        nbhd_graph.add_nodes_from(range(len(safe_nbhds)))
        for idx, safe_nbhd in enumerate(safe_nbhds):
            for point in safe_nbhd:
                nbhd_graph.add_edge(idx, point)
        self.dimension_ = max(nx.algorithms.clique.find_cliques(nbhd_graph), key=len)

    @staticmethod
    def _visible_neighbors(X, idx, nbhd, radial_dists):
        if len(nbhd) == 1:
            return nbhd, radial_dists
        neighbors = np.take(X, nbhd, 0)
        vectors = neighbors - X[idx]

        visibility = []
        for i, vector in enumerate(vectors):
            visible = True
            if i != 0:
                for j, other_vector in enumerate(vectors):
                    if j == 0 or j == i:
                        continue
                    onward_vector = vector - other_vector
                    if np.dot(onward_vector, other_vector) >= 0:
                        visible = False
                        break
            visibility.append(visible)
        visibility = np.array(visibility)

        visible_nbhd = nbhd[visibility]
        radial_dists = radial_dists[visibility]

        return visible_nbhd, radial_dists

    @staticmethod
    def _safe_neighbors(X, nbhd, radial_dist):
        neighbors = np.take(X, nbhd, axis=0)
        incremental_dims = [0]
        for i, _ in enumerate(neighbors):
            if i==0:
                continue
            data = neighbors[0:i+1, :]
            incremental_dims.append(lPCA(ver="FO", alphaFO=0.05).fit_transform(data))
        jumps = np.zeros(shape=(len(incremental_dims),))
        for i, dim in enumerate(incremental_dims):
            if i == 0:
                continue
            if dim - incremental_dims[i - 1] > 0:
                jumps[i] = radial_dist[i] - radial_dist[i - 1]
        return jumps
        # compare to average jumps

    def _calculate_jumps(self, X, idx, nbhd, radial_dists):
        visible_nbhd, radial_dists = LinZha._visible_neighbors(
            X, idx, nbhd, radial_dists
        )
        jumps = LinZha._safe_neighbors(X, visible_nbhd, radial_dists)
        return visible_nbhd, radial_dists, jumps
