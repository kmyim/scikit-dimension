import numpy as np
from .._commonfuncs import FlexNbhdEstimator
from sklearn.utils.parallel import Parallel, delayed
from joblib import effective_n_jobs


class CDim(FlexNbhdEstimator):
    """
    Conical dimension method with either knn or epsilon neighbourhood

    Yang et al.
    Conical dimension as an intrinsic dimension estimator and its applications
    https://doi.org/10.1137/1.9781611972771.16
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

        if nbhd_type not in ["eps", "knn"]:
            raise ValueError("Neighbourhood type should either be knn or eps.")

        if effective_n_jobs(n_jobs) > 1:
            with Parallel(n_jobs=n_jobs) as parallel:
                # Asynchronously apply the `fit` function to each data point and collect the results
                results = parallel(
                    delayed(self._local_cdim)(X, idx, nbhd)
                    for idx, nbhd in enumerate(nbhd_indices)
                )
            self.dimension_pw_ = np.array(results)
        else:
            self.dimension_pw_ = np.array(
                [
                    self._local_cdim(X, idx, nbhd)
                    for idx, nbhd in enumerate(nbhd_indices)
                ]
            )

    @staticmethod
    def _local_cdim(X, idx, nbhd):
        if len(nbhd) == 0:
            return 0
        neighbors = np.take(X, nbhd, 0)
        vectors = neighbors - X[idx]
        norms = np.linalg.norm(vectors, axis=1)
        vectors = np.take(vectors, np.argsort(norms), axis=0)
        vectors = vectors * np.sign(np.tensordot(vectors, vectors[0], axes=1))[:, None]
        T = np.matmul(vectors, vectors.T)
        label_set = [[i] for i, _ in enumerate(nbhd)]
        de = 0
        while len(label_set) > 0:
            de += 1
            new_label_set = []
            for label in label_set:
                for i, _ in enumerate(nbhd):
                    if i in label:
                        continue
                    for vec_idx in label:
                        if T[i][vec_idx] >= 0:
                            break
                    else:
                        new_label = label + [i]
                        new_label_set.append(new_label)
            label_set = new_label_set
        return de
