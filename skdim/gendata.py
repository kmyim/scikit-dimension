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
from sklearn.utils.validation import check_random_state
from ._commonfuncs import hyperBall


def hyperSphere(n, d, center=[], random_state=None):
    """
    Generates a sample from a uniform distribution on an hypersphere surface
    """
    random_state = check_random_state(random_state)
    vec = random_state.randn(n, d)
    vec /= np.linalg.norm(vec, axis=1)[:, None]
    return vec

def hyperTwinPeaks(n, d=2, height=1, random_state=None):
    """ 
    Translated from Kerstin Johnsson's R package intrinsicDimension
    """
    random_state = check_random_state(random_state)
    base_coord = random_state.uniform(size=(n, d))
    _height = height * np.prod(np.sin(2 * np.pi * base_coord), axis=1, keepdims=1)
    return np.hstack((base_coord, _height))

def lineDiskBall(n, random_state=None):
    """ 
    Generates a sample from a uniform distribution on a line, an oblong disk and an oblong ball
    Translated from ldbl function in Hideitsu Hino's package
    """
    random_state = check_random_state(random_state)

    line = np.hstack(
        (
            np.repeat(0, 5 * n)[:, None],
            np.repeat(0, 5 * n)[:, None],
            random_state.uniform(-0.5, 0, size=5 * n)[:, None],
        )
    )
    disc = np.hstack(
        (
            random_state.uniform(-1, 1, (13 * n, 2)),
            np.zeros(13 * n)[:, None],
        )
    )
    disc = disc[~(np.sqrt(np.sum(disc ** 2, axis=1)) > 1), :]
    disc = disc[:, [0, 2, 1]]
    disc[:, 2] = disc[:, 2] - min(disc[:, 2]) + max(line[:, 2])

    fb = random_state.uniform(-0.5, 0.5, size=(n * 100, 3))
    rmID = np.where(np.sqrt(np.sum(fb ** 2, axis=1)) > 0.5)[0]

    if len(rmID) > 0:
        fb = fb[~(np.sqrt(np.sum(fb ** 2, axis=1)) > 0.5), :]

    fb = np.hstack((fb[:, :2], fb[:, [2]] + 0.5))
    fb[:, 2] = fb[:, 2] - min(fb[:, 2]) + max(disc[:, 2])

    #     if _sorted:
    #         fb = fb[order(fb[:, 2]),:]

    line2 = np.hstack(
        (
            np.repeat(0, 5 * n)[:, None],
            np.repeat(0, 5 * n)[:, None],
            random_state.uniform(-0.5, 0, size=5 * n)[:, None],
        )
    )
    line2[:, 2] = line2[:, 2] - min(line2[:, 2]) + max(fb[:, 2])
    lineID = np.repeat(1, len(line))
    discID = np.repeat(2, len(disc))
    fbID = np.repeat(3, len(fb))
    line2ID = np.repeat(1, len(line2))
    x = np.vstack((line, disc, fb, line2))
    useID = np.sort(random_state.choice(len(x), n, replace=False))
    x = x[useID, :]

    return x, np.concatenate((lineID, discID, fbID, line2ID), axis=0)[useID]

def swissRoll3Sph(n_swiss, n_sphere, a=1, b=2, nturn=1.5, h=4, random_state=None):
    """
    Generates a sample from a uniform distribution on a Swiss roll-surface, 
    possibly together with a sample from a uniform distribution on a 3-sphere
    inside the Swiss roll. Translated from Kerstin Johnsson's R package intrinsicDimension

    Parameters
    ----------

    Ns : int 
        Number of data points on the Swiss roll.

    Nsph : int
        Number of data points on the 3-sphere.

    a : int or float, default=1
        Minimal radius of Swiss roll and radius of 3-sphere.

    b : int or float, default=2
        Maximal radius of Swiss roll.

    nturn : int or float, default=1.5
        Number of turns of the surface. 

    h : int or float, default=4
        Height of Swiss roll.

    Returns
    -------
    
    np.array, (npoints x ndim)
    """
    random_state = check_random_state(random_state)

    if n_swiss > 0:
        omega = 2 * np.pi * nturn
        dl = lambda r: np.sqrt(b ** 2 + omega ** 2 * (a + b * r) ** 2)
        ok = np.zeros(1)
        while sum(ok) < n_swiss:
            r_samp = random_state.uniform(size=3 * n_swiss)
            ok = random_state.uniform(size=3 * n_swiss) < dl(r_samp) / dl(1)

        r_samp = r_samp[ok][:n_swiss]
        x = (a + b * r_samp) * np.cos(omega * r_samp)
        y = (a + b * r_samp) * np.sin(omega * r_samp)
        z = random_state.uniform(-h, h, size=n_swiss)
        w = np.zeros(n_swiss)

    else:
        x = y = z = w = np.array([])

    if n_sphere > 0:
        sph = hyperSphere(n_sphere, 4, random_state=random_state) * a
        x = np.concatenate((x, sph[:, 0]))
        y = np.concatenate((y, sph[:, 1]))
        z = np.concatenate((z, sph[:, 2]))
        w = np.concatenate((w, sph[:, 3]))

    return np.hstack((x[:, None], y[:, None], z[:, None], w[:, None]))


class BenchmarkManifolds:
    """
    Generates a commonly used benchmark set of synthetic manifolds with known intrinsic dimension described in [1], [2]
    -----------
    Attributes

   type_noise : str, 'uniform' or 'gaussian'
        Type of noise to generate
    d : int, default=1
        For ver ='a', any value of d is possible,  for ver ='b', only d = 1 is supported.
    -----------
    Returns

    dimension_ : int
        Intrinsic dimension of the dataset
    random_state : None, int or np.random.RandomState
        Fix random number generation

    -----------
    References

    [1] Campadelli et al., Intrinsic Dimension Estimation: Relevant Techniques and 
    a Benchmark Framework, https://doi.org/10.1155/2015/759567
    [2] M. Hein and J.-Y. Audibert, Intrinsic dimensionality estimation of submanifolds in Euclidean space, Proceedings of the 22nd Internatical Conference on Machine Learning (ICML), 289--296. (Eds.) L. de Raedt and S. Wrobel (2005). 
    """

    # class modified and adapted from https://github.com/stat-ml/GeoMLE
    # Original licence citation:
    # MIT License
    #
    # Copyright (c) 2019 Mokrov Nikita, Marina Gomtsyan, Maxim Panov and Yury Yanovich
    #
    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included in all
    # copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    # SOFTWARE.

    def __init__(self, random_state: int = None, noise_type: str = "uniform"):

        self.random_state = check_random_state(random_state)

        self.dict_truth = {
            "M1_Sphere":         (10, 11),              
            "M2_Affine_3to5":    (3, 5),
            "M3_Nonlinear_4to6": (4, 6),
            "M4_Nonlinear":      (4, 8),
            "M5a_Helix1d":       (1, 3),
            "M5b_Helix2d":       (2, 3),
            "M6_Nonlinear":      (6, 36),
            "M7_Roll":           (2, 3),
            "M8_Nonlinear":      (12,72),
            "M9_Affine":         (20, 20),
            "M10a_Cubic":        (10, 11),
            "M10b_Cubic":        (17, 18),
            "M10c_Cubic":        (24, 25),
            "M10d_Cubic":        (70, 71),
            "M11_Moebius":       (2, 3),
            "M12_Norm":          (20, 20),
            "M13a_Scurve":       (1, 13),
            "M13b_Spiral":       (1, 13),
            "Mbeta":             (10,40),
            "Mn1_Nonlinear":     (18, 72),
            "Mn2_Nonlinear":     (24, 96),
            "Mp1_Paraboloid":    (3, 12),
            "Mp2_Paraboloid":     (6, 21),
            "Mp3_Paraboloid":    (9, 30),
        }

        self.dict_gen = {
            # synthetic data
            "M1_Sphere": _gen_sphere_data,
            "M2_Affine_3to5": _gen_affine3_5_data,
            "M3_Nonlinear_4to6": _gen_nonlinear4_6_data,
            "M4_Nonlinear": _gen_nonlinear_data,
            "M5a_Helix1d": _gen_helix1_data,
            "M5b_Helix2d": _gen_helix2_data,
            "M6_Nonlinear": _gen_nonlinear_data,
            "M7_Roll": _gen_roll_data,
            "M8_Nonlinear": _gen_nonlinear_data,
            "M9_Affine": _gen_affine_data,
            "M10a_Cubic": _gen_cubic_data,
            "M10b_Cubic": _gen_cubic_data,
            "M10c_Cubic": _gen_cubic_data,
            "M10d_Cubic": _gen_cubic_data,
            "M11_Moebius": _gen_moebius_data,
            "M12_Norm": _gen_norm_data,
            "M13a_Scurve": _gen_scurve_data,
            "M13b_Spiral": _gen_spiral_data,
            "Mbeta": _gen_campadelli_beta_data
            "Mn1_Nonlinear": _gen_campadelli_n_data,
            "Mn2_Nonlinear": _gen_campadelli_n_data,
            "Mp1_Paraboloid": _gen_paraboloid_data,
            "Mp2_Paraboloid": _gen_paraboloid_data,
            "Mp3_Paraboloid": _gen_paraboloid_data
        }

    def generate(
        self,
        name: str = 'all',
        n: int = 2500,
        dim: int = None,
        d: int = None,
        noise: float = 0.0,
    ):

        # Generates all datasets. A ground truth dict of intrinsic dimension and embedding dimension is in BenchmarkManifolds.dict_truth.keys()
        #
        # Parameters:
        # --------------------
        # n: int
        #     The number of sample points
        # dim: int
        #     If generating a single dataset, choose the embedding dimension. Note that some datasets have restrictions on the chosen embedding dimension
        # d: int
        #     If generating a single dataset, choose the intrinsic dimension. Note that some datasets have restrictions on the chosen intrinsic dimension
        # noise: float, optional(default=0.0)
        #     The value of noise in data
        #
        # Returns:
        # --------------------
        # data: a dict of np.arrays or a single np.array with shape (n, dim)
        #    Generated data
        
        if self.noise_type == "normal":
            self.gen_noise = self.random_state.randn
        if self.noise_type == "uniform":
            self.gen_noise = lambda n, dim: self.random_state.rand(n, dim) - 0.5

        dict_data = {}
        if name == 'all':
            for k,(d,dim) in self.dict_truth.items():
                data = self.dict_gen[k](n=n, dim=dim, d=d)
                noise = self.gen_noise(n, dim) * noise
                dict_data[k] = data + noise
            return dict_data

        elif name in self.dict_truth.keys()
            data = self.dict_gen[k](n=n, dim=dim, d=d)
            noise = self.gen_noise(n, dim) * noise
            return data + noise

    def _gen_spiral_data(self, n, dim=3, d=1):
        assert d < dim
        assert d == 1
        assert dim >= 3

        t = 10 * np.pi * self.random_state.rand(n)
        data = np.vstack([100 * np.cos(t), 100 * np.sin(t), t, np.zeros((dim - 3, n))]).T
        assert data.shape == (n, dim)
        return data


    def _gen_helix1_data(self, n, dim=3, d=1):
        assert d < dim
        assert d == 1
        assert dim >= 3

        t = 2 * np.pi / n + self.random_state.rand(n) * 2 * np.pi
        data = np.vstack(
            [
                (2 + np.cos(8 * t)) * np.cos(t),
                (2 + np.cos(8 * t)) * np.sin(t),
                np.sin(8 * t),
                np.zeros((dim - 3, n)),
            ]
        ).T
        assert data.shape == (n, dim)
        return data


    def _gen_helix2_data(self, n, dim=3, d=2):
        assert d < dim
        assert d == 2
        assert dim >= 3
        r = 10 * np.pi * self.random_state.rand(n)
        p = 10 * np.pi * self.random_state.rand(n)
        data = np.vstack([r * np.cos(p), r * np.sin(p), 0.5 * p, np.zeros((dim - 3, n))]).T
        assert data.shape == (n, dim)
        return data


    def _gen_helicoid_data(self, n, dim=3, d=2):
        assert d <= dim
        assert d == 2
        assert dim >= 3

        u = 2 * np.pi / n + self.random_state.rand(n) * 2 * np.pi
        v = 5 * np.pi * self.random_state.rand(n)
        data = np.vstack([np.cos(v), np.sin(v) * np.cos(v), u, np.zeros((dim - 3, n))]).T
        assert data.shape == (n, dim)
        return data


    def _gen_roll_data(self, n, dim=3, d=2):
        assert d < dim
        assert dim >= 3
        assert d == 2

        t = 1.5 * np.pi * (1 + 2 * self.random_state.rand(n))
        p = 21 * self.random_state.rand(n)

        data = np.vstack([t * np.cos(t), p, t * np.sin(t), np.zeros((dim - d - 1, n))]).T
        assert data.shape == (n, dim)
        return data


    def _gen_scurve_data(self, n, dim=3, d=2):
        assert d < dim
        assert dim >= 3
        assert d == 2

        t = 3 * np.pi * (self.random_state.rand(n) - 0.5)
        p = 2.0 * self.random_state.rand(n)

        data = np.vstack(
            [np.sin(t), p, np.sign(t) * (np.cos(t) - 1), np.zeros((dim - d - 1, n))]
        ).T
        assert data.shape == (n, dim)
        return data


    def _gen_sphere_data(self, n, dim, d):
        assert d < dim

        V = self.random_state.randn(n, d + 1)
        data = np.hstack(
            [V / np.sqrt((V ** 2).sum(axis=1))[:, None], np.zeros((n, dim - d - 1))]
        )
        assert data.shape == (n, dim)
        return data


    def _gen_norm_data(self, n, dim, d):
        assert d <= dim

        norm_xyz = self.random_state.multivariate_normal(np.zeros(d), np.identity(d), n)
        data = np.hstack([norm_xyz, np.zeros((n, dim - d))])
        assert data.shape == (n, dim)
        return data


    def _gen_uniform_data(self, n, dim, d):
        assert d <= dim
        uniform_xyz = self.random_state.uniform(size=(n, d))
        data = np.hstack([uniform_xyz, np.zeros((n, dim - d))])
        assert data.shape == (n, dim)
        return data


    def _gen_cubic_data(self, n, dim, d):
        assert d < dim
        cubic_data = np.array([[]] * (d + 1))
        for i in range(d + 1):
            n_once = int(n / (2 * (d + 1)) + 1)
            # 1st side
            data_once = self.random_state.rand(d + 1, n_once)
            data_once[i] = 0
            cubic_data = np.hstack([cubic_data, data_once])
            # 2nd side
            data_once = self.random_state.rand(d + 1, n_once)
            data_once[i] = 1
            cubic_data = np.hstack([cubic_data, data_once])
        cubic_data = cubic_data.T[:n]
        data = np.hstack([cubic_data, np.zeros((n, dim - d - 1))])
        assert data.shape == (n, dim)
        return data


    def _gen_moebius_data(self, n, dim=3, d=2):
        assert dim == 3
        assert d == 2

        phi = self.random_state.rand(n) * 2 * np.pi
        rad = self.random_state.rand(n) * 2 - 1
        data = np.vstack(
            [
                (1 + 0.5 * rad * np.cos(5.0 * phi)) * np.cos(phi),
                (1 + 0.5 * rad * np.cos(5.0 * phi)) * np.sin(phi),
                0.5 * rad * np.sin(5.0 * phi),
            ]
        ).T

        assert data.shape == (n, dim)
        return data


    def _gen_affine_data(self, n, dim, d):
        assert dim >= d

        p = self.random_state.rand(d, n) * 5 - 2.5
        v = np.eye(dim, d)
        #     v = np.random.randint(0, 10, (dim, d))
        data = v.dot(p).T

        assert data.shape == (n, dim)
        return data


    def _gen_affine3_5_data(self, n, dim=5, d=3):
        assert dim == 5
        assert d == 3

        p = 4 * self.random_state.rand(d, n)
        A = np.array(
            [
                [1.2, -0.5, 0],
                [0.5, 0.9, 0],
                [-0.5, -0.2, 1],
                [0.4, -0.9, -0.1],
                [1.1, -0.3, 0],
            ]
        )
        b = np.array([[3, -1, 0, 0, 8]]).T
        data = A.dot(p) + b
        data = data.T

        assert data.shape == (n, dim)
        return data


    def _gen_nonlinear4_6_data(self, n, dim=6, d=4):
        assert dim == 6
        assert d == 4

        p0, p1, p2, p3 = self.random_state.rand(d, n)
        data = np.vstack(
            [
                p1 ** 2 * np.cos(2 * np.pi * p0),
                p2 ** 2 * np.sin(2 * np.pi * p0),
                p1 + p2 + (p1 - p3) ** 2,
                p1 - 2 * p2 + (p0 - p3) ** 2,
                -p1 - 2 * p2 + (p2 - p3) ** 2,
                p0 ** 2 - p1 ** 2 + p2 ** 2 - p3 ** 2,
            ]
        ).T

        assert data.shape == (n, dim)
        return data


    def _gen_nonlinear_data(self, n, dim, d):
        assert dim >= d
        m = int(dim / (2 * d))
        assert dim == 2 * m * d

        p = self.random_state.rand(d, n)
        F = np.zeros((2 * d, n))
        F[0::2, :] = np.cos(2 * np.pi * p)
        F[1::2, :] = np.sin(2 * np.pi * p)
        R = np.zeros((2 * d, n))
        R[0::2, :] = np.vstack([p[1:], p[0]])
        R[1::2, :] = np.vstack([p[1:], p[0]])
        D = (R * F).T
        data = np.hstack([D] * m)

        assert data.shape == (n, dim)
        return data


    def _gen_paraboloid_data(self, n, dim, d):
        assert dim == 3 * (d + 1)
        random_state = check_random_state(random_state)

        E = self.random_state.exponential(1, (d + 1, n))
        X = ((1 + E[1:] / E[0]) ** -1).T
        X = np.hstack([X, (X ** 2).sum(axis=1)[:, np.newaxis]])
        data = np.hstack([X, np.sin(X), X ** 2])

        assert data.shape == (n, dim)
        return data

    def _gen_campadelli_n_data(self, n, dim, d):
        assert dim == d*4

        #Generate points drawn from a uniform distribution
        X = self.random_state.random(size=(n, d))
        temp1 = np.zeros((n, d))
        temp2 = np.zeros((n, d))

        #Extend the embedding dimensionality: 
        for i in range(d):
            temp1[:,i] = np.tan(X[:,i]*np.cos(X[:,d-1-i]));
            temp2[:,i] = np.arctan(X[:,d-1-i]*np.sin(X[:,i]));

        #Create the final dataset:
        data = np.concatenate([temp1,temp2,temp1,temp2],axis=1)
        return data

    def _gen_campadelli_beta_data(self, n, dim, d, alpha=10, beta=0.5):
        assert dim == d*4

        # Function to generate point drawn from a beta distribution
        X = self.random_state.beta(a = alpha, b = beta, size=((n,d)))  

        temp1 = np.zeros((n, d))
        temp2 = np.zeros((n, d))

        # Extend the embedding dimensionality: 
        for i in range(d):
            temp1[:,i] = X[:,i]*np.sin(np.cos(2*np.pi*(X[:,i])))
            temp2[:,i] = X[:,i]*np.cos(np.sin(2*np.pi*(X[:,i])))

        # Create the final dataset:
        data = np.concatenate([temp1,temp2,temp1,temp2],axis=1)
        return data