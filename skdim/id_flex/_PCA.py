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
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_array
from .._commonfuncs import GlobalEstimator
from .._commonfuncs import FlexNbhdEstimator


class lPCA(FlexNbhdEstimator):
    # SPDX-License-Identifier: MIT, 2017 Kerstin Johnsson [IDJohnsson]_
    """Intrinsic dimension estimation using the PCA algorithm. [Cangelosi2007]_ [Fan2010]_ [Fukunaga2010]_ [IDJohnsson]_

    Version 'FO' (Fukunaga-Olsen) returns eigenvalues larger than alphaFO times the largest eigenvalue.\n
    Version 'Fan' is the method by Fan et al.\n
    Version 'maxgap' returns the position of the largest relative gap in the sequence of eigenvalues.\n
    Version 'ratio' returns the number of eigenvalues needed to retain at least alphaRatio of the variance.\n
    Version 'participation_ratio' returns the number of eigenvalues given by PR=sum(eigenvalues)^2/sum(eigenvalues^2)\n
    Version 'Kaiser' returns the number of eigenvalues above average (the average eigenvalue is 1)\n
    Version 'broken_stick' returns the number of eigenvalues above corresponding values of the broken stick distribution\n

    Parameters
    ----------
    ver: str, default='FO'
        Version. Possible values: 'FO', 'Fan', 'maxgap','ratio', 'Kaiser', 'broken_stick'.
    alphaRatio: float in (0,1)
        Only for ver = 'ratio'. ID is estimated to be 
        the number of principal components needed to retain at least alphaRatio of the variance.
    alphaFO: float in (0,1)
        Only for ver = 'FO'. An eigenvalue is considered significant 
        if it is larger than alpha times the largest eigenvalue.
    alphaFan: float
        Only for ver = 'Fan'. The alpha parameter (large gap threshold).
    betaFan: float
        Only for ver = 'Fan'. The beta parameter (total covariance threshold).
    PFan: float
        Only for ver = 'Fan'. Total covariance in non-noise.
    verbose: bool, default=False
    fit_explained_variance: bool, default=False
        If True, lPCA.fit(X) expects as input 
        a precomputed explained_variance vector: X = sklearn.decomposition.PCA().fit(X).explained_variance_
    
    Attributes
    ----------
    gap_pw_:
        Ratio of each PC's explained variance (except the last)
        with the following PC's explained variance at each point
    """

    def __init__(
        self,
        ver="FO",
        alphaRatio=0.05,
        alphaFO=0.05,
        alphaFan=10,
        betaFan=0.8,
        PFan=0.95,
        verbose=True,
        fit_explained_variance=False,
    ):
        self.ver = ver
        self.alphaRatio = alphaRatio
        self.alphaFO = alphaFO
        self.alphaFan = alphaFan
        self.betaFan = betaFan
        self.PFan = PFan
        self.verbose = verbose
        self.fit_explained_variance = fit_explained_variance


    def _fit(self, X, nbhd_indices, nbhd_type, metric, radial_dists, radius = 1.0, n_neighbors = 5,):

        if nbhd_type not in ['eps', 'knn']: raise ValueError('Neighbourhood type should either be knn or eps.')

        if self.fit_explained_variance:
            X = check_array(X, ensure_2d=False, ensure_min_samples=2)
        else:
            X = check_array(X, ensure_min_samples=2, ensure_min_features=2)

        dims_and_gaps = [self._pcaLocalDimEst(X[nbhd]) for nbhd in nbhd_indices]

        self.dimension_pw_, self.gap_pw_ = np.array([list_item[0] for list_item in dims_and_gaps]), np.array([list_item[1] for list_item in dims_and_gaps])


    def _pcaLocalDimEst(self, X):
        if self.fit_explained_variance:
            explained_var = X
        else:
            pca = PCA().fit(X)
            self.explained_var_ = explained_var = pca.explained_variance_

        if self.ver == "FO":
            return self._FO(explained_var)
        elif self.ver == "Fan":
            return self._fan(explained_var)
        elif self.ver == "maxgap":
            return self._maxgap(explained_var)
        elif self.ver == "ratio":
            return self._ratio(explained_var)
        elif self.ver == "participation_ratio":
            return self._participation_ratio(explained_var)
        elif self.ver == "Kaiser":
            return self._Kaiser(explained_var)
        elif self.ver == "broken_stick":
            return self._broken_stick(explained_var)

    def _FO(self, explained_var):
        de = sum(explained_var > (self.alphaFO * explained_var[0]))
        gaps = explained_var[:-1] / explained_var[1:]
        return de, gaps


    @staticmethod
    def _maxgap(explained_var):
        gaps = explained_var[:-1] / explained_var[1:]
        de = np.nanargmax(gaps) + 1
        return de, gaps


    def _ratio(self, explained_var):
        sumexp = np.cumsum(explained_var)
        sumexp_norm = sumexp / np.max(sumexp)
        de = sum(sumexp_norm < self.alphaRatio) + 1
        gaps = explained_var[:-1] / explained_var[1:]
        return de, gaps


    def _participation_ratio(self, explained_var):
        PR = sum(explained_var) ** 2 / sum(explained_var ** 2)
        de = PR
        gaps = explained_var[:-1] / explained_var[1:]
        return de, gaps


    def _fan(self, explained_var):
        r = np.where(np.cumsum(explained_var) / sum(explained_var) > self.PFan)[0][0]
        sigma = np.mean(explained_var[r:])
        explained_var -= sigma
        gaps = explained_var[:-1] / explained_var[1:]
        de = 1 + np.min(
            np.concatenate(
                (
                    np.where(gaps > self.alphaFan)[0],
                    np.where(
                        (np.cumsum(explained_var) / sum(explained_var)) > self.betaFan
                    )[0],
                )
            )
        )
        return de, gaps


    def _Kaiser(self, explained_var):
        de = sum(explained_var > np.mean(explained_var))
        gaps = explained_var[:-1] / explained_var[1:]
        return de, gaps


    @staticmethod
    def _brokenstick_distribution(dim):
        distr = np.zeros(dim)
        for i in range(dim):
            for j in range(i, dim):
                distr[i] = distr[i] + 1 / (j + 1)
            distr[i] = distr[i] / dim
        return distr

    def _broken_stick(self, explained_var):
        bs = self._brokenstick_distribution(dim=len(explained_var))
        gaps = explained_var[:-1] / explained_var[1:]
        de = 0
        explained_var_norm = explained_var / np.sum(explained_var)
        for i in range(len(explained_var)):
            if bs[i] > explained_var_norm[i]:
                de = i + 1
                break
        return de, gaps
