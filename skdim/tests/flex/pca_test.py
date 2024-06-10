import pytest
import numpy as np
import skdim.id_flex
from sklearn.datasets import make_swiss_roll

VERSIONS = ['FO', 'Fan', 'maxgap','ratio', 'Kaiser', 'broken_stick']


@pytest.fixture
def data():
    X = np.zeros((30, 10))
    X[:, :5] = skdim.datasets.hyperBall(n=30, d=5, radius=1, random_state=0)
    return X

@pytest.mark.parametrize("ver", VERSIONS)
def test_lpca_results(data, ver):
    assert skdim.id_flex.lPCA(ver=ver).fit(data, nbhd_type="eps", radius=2.0).dimension_ == 5


@pytest.mark.parametrize("ver", VERSIONS)
def test_on_swiss_roll(ver):
    np.random.seed(782)
    swiss_roll_dat = make_swiss_roll(1000)
    pca = skdim.id_flex.lPCA(ver=ver)
    estim_dim = pca.fit_transform(swiss_roll_dat[0], nbhd_type='knn',
                                  n_neighbors=20)
    assert 2.0 == pytest.approx(estim_dim, 0.1)


def test_exception_is_raised_when_neighbourhoods_empty():
    np.random.seed(123)
    dist_matrix = __generate_distance_matrix(10, 12)
    mle = skdim.id_flex.MLE_basic()
    with pytest.raises(ValueError):
        mle.fit_transform(dist_matrix, nbhd_type='eps', metric='precomputed',
                          radius=5)


def test_when_eps_and_knn_almost_equivalent():
    pca = skdim.id_flex.lPCA()
    rectangle = np.zeros((4, 4))
    rectangle[1:3, 1] = 2
    rectangle[:2, 0] = 1
    knn_estim = pca.fit_transform(rectangle, nbhd_type='knn', n_neighbors=2)
    eps_estim = pca.fit_transform(rectangle, nbhd_type='eps', radius=2)
    assert knn_estim * 2.0 == pytest.approx(eps_estim)
