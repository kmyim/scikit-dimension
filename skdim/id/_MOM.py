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
from .._commonfuncs import LocalEstimator


class MOM(LocalEstimator):
    """
    Intrinsic dimension estimation using the Method Of Moments algorithm.

    Attributes
    ----------
    None
    
    References
    ----------
    Code translated from the original implementation by Miloš Radovanović (https://perun.pmf.uns.ac.rs/radovanovic/tle/).

     L.  Amsaleg,  O.  Chelly,  T.  Furon,  S.  Girard,  M.  E.Houle,  K.  Kawarabayashi,  and  M.  Nett. Extreme-value-theoretic estimation of local intrinsic dimensionality.DAMI, 32(6):1768–1805, 2018.
    """

    def _fit(self, **kwargs):
        self.dimension_pw_ = self._mom(kwargs["dists"])

    def _mom(self, dists):
        # dists - nearest-neighbor distances (k x 1, or k x n), sorted
        w = dists[:, -1]
        m1 = np.sum(dists, axis=1) / dists.shape[1]
        ID = -m1 / (m1 - w)
        return ID
