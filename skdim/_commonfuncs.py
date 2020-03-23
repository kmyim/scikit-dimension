#
# MIT License
#
# Copyright (c) 2020 Jonathan Bac
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
#
import numpy as np
import itertools
import numbers
import multiprocessing as mp
from sklearn.utils.validation import check_random_state
from sklearn.neighbors import NearestNeighbors
from scipy.special import gammainc
from inspect import getmembers, isclass
import skdim


def get_estimators():
    local_class_list = [o[1]
                        for o in getmembers(skdim.local_id) if isclass(o[1])]
    global_class_list = [o[1]
                         for o in getmembers(skdim.global_id) if isclass(o[1])]

    local_estimators = dict(
        zip([str(e).split('.')[-1][:-2] for e in local_class_list], local_class_list))
    global_estimators = dict(
        zip([str(e).split('.')[-1][:-2] for e in global_class_list], global_class_list))

    return local_estimators, global_estimators


def indComb(NN):
    pt1 = np.tile(range(NN), NN)
    pt2 = np.repeat(range(NN), NN)

    un = pt1 > pt2

    pt1 = pt1[un]
    pt2 = pt2[un]

    return pt1, pt2, np.hstack((pt2[:, None], pt1[:, None]))


def indnComb(NN, n):
    if (n == 1):
        return(np.arange(NN).reshape((-1, 1)))
    prev = indnComb(NN, n-1)
    lastind = prev[:, -1]
    ind_cf1 = np.repeat(lastind, NN)
    ind_cf2 = np.tile(np.arange(NN), len(lastind))
    #ind_cf2 = np.arange(NN)
    # for i in range(len(lastind)-1):
    #    ind_cf2 = np.concatenate((ind_cf2,np.arange(NN)))
    new_ind = np.where(ind_cf1 < ind_cf2)[0]
    new_ind1 = ((new_ind - 1) // NN)
    new_ind2 = new_ind % NN
    new_ind2[new_ind2 == 0] = NN
    return np.hstack((prev[new_ind1, :], np.arange(NN)[new_ind2].reshape((-1, 1))))


def efficient_indnComb(n, k, random_generator_):
    '''
    memory-efficient indnComb:
    uniformly takes 5000 samples from itertools.combinations(n,k)
    '''
    ncomb = binom_coeff(n, k)
    pop = itertools.combinations(range(n), k)
    targets = set(random_generator_.choice(
        ncomb, min(ncomb, 5000), replace=False))
    return np.array(list(itertools.compress(pop, map(targets.__contains__, itertools.count()))))


def lens(vectors):
    return np.sqrt(np.sum(vectors**2, axis=1))


def randball(n_points, n_dim, radius, center=[], random_state=None):
    random_state_ = check_random_state(random_state)
    if center == []:
        center = np.array([0]*n_dim)
    r = radius
    x = random_state_.normal(size=(n_points, n_dim))
    ssq = np.sum(x**2, axis=1)
    fr = r*gammainc(n_dim/2, ssq/2)**(1/n_dim)/np.sqrt(ssq)
    frtiled = np.tile(fr.reshape(n_points, 1), (1, n_dim))
    p = center + np.multiply(x, frtiled)
    return p


def proxy(tup):
    function, X, Dict = tup
    return function(X, **Dict)


def get_nn(X, k, n_jobs=1):
    neigh = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs)
    neigh.fit(X)
    dists, inds = neigh.kneighbors(return_distance=True)
    return dists, inds


def asPointwise(data, class_instance, precomputed_knn=None, n_neighbors=100, n_jobs=1):
    '''Use a global estimator as a pointwise one by creating kNN neighborhoods'''
    if precomputed_knn is not None:
        knn = precomputed_knn
    else:
        _, knn = get_nn(data, k=n_neighbors, n_jobs=n_jobs)

    if n_jobs > 1:
        pool = mp.Pool(n_jobs)
        results = pool.map(
            class_instance.fit, [data[i, :] for i in knn])
        pool.close()
        return results
    else:
        return [class_instance.fit(data[i, :]).dimension_ for i in knn]


def binom_coeff(n, k):
    '''
    Taken from : https://stackoverflow.com/questions/26560726/python-binomial-coefficient
    Compute the number of ways to choose $k$ elements out of a pile of $n.$

    Use an iterative approach with the multiplicative formula:
    $$\frac{n!}{k!(n - k)!} =
    \frac{n(n - 1)\dots(n - k + 1)}{k(k-1)\dots(1)} =
    \prod_{i = 1}^{k}\frac{n + 1 - i}{i}$$

    Also rely on the symmetry: $C_n^k = C_n^{n - k},$ so the product can
    be calculated up to $\min(k, n - k).$

    :param n: the size of the pile of elements
    :param k: the number of elements to take from the pile
    :return: the number of ways to choose k elements out of a pile of n
    '''

    # When k out of sensible range, should probably throw an exception.
    # For compatibility with scipy.special.{comb, binom} returns 0 instead.
    if k < 0 or k > n:
        return 0

    if k == 0 or k == n:
        return 1

    total_ways = 1
    for i in range(min(k, n - k)):
        total_ways = total_ways * (n - i) // (i + 1)

    return total_ways


def check_random_generator(seed):
    """Turn seed into a numpy.random._generator.Generator' instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.default_rng()
    if isinstance(seed, numbers.Integral):
        return np.random.default_rng(seed)
    if isinstance(seed, np.random._generator.Generator):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random._generator.Generator'
                     ' instance' % seed)
