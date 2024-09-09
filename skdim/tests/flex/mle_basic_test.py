import pytest
import numpy as np
import skdim.id_flex
from sklearn.datasets import make_swiss_roll
import skdim.errors

def __generate_distance_matrix(size, threshold=10, maximum=100):
    # Generate a random distance matrix with values between 0 and 1sklea
    distance_matrix = np.random.rand(size, size)
    
    # Scale the values to fit the desired range (e.g., 0 to 100)
    distance_matrix = distance_matrix * maximum
    
    # Ensure that each value is greater than the threshold
    distance_matrix = np.maximum(distance_matrix, threshold)
    
    # Fill diagonal with zeros (optional, depends on your use case)
    np.fill_diagonal(distance_matrix, 0)
    
    return distance_matrix

def test_on_swiss_roll():
    np.random.seed(782)
    swiss_roll_dat = make_swiss_roll(1000)
    mle = skdim.id_flex.MLE_basic(nbhd_type = 'knn', n_neighbors=20)
    estim_dim = mle.fit_transform(swiss_roll_dat[0])
    # The expected value is taken from Levina and Bickel's paper
    assert 2.1 == pytest.approx(estim_dim, 0.1)

def test_on_swiss_roll_hmean():
    np.random.seed(782)
    swiss_roll_dat = make_swiss_roll(1000)
    mle = skdim.id_flex.MLE_basic(nbhd_type = 'knn', n_neighbors=20, comb='hmean')
    estim_dim = mle.fit_transform(swiss_roll_dat[0])
    # No standard expected result
    assert 2.0 == pytest.approx(estim_dim, 0.1)

# this might be too stringent a test?
def test_on_equal_distances():
    SIZE = 5
    distances = np.full((SIZE, SIZE), 0.75)
    for i in range(SIZE):
        distances[i,i] = 0.0
    mle = skdim.id_flex.MLE_basic( metric="precomputed", n_neighbors=3)
    with pytest.raises(skdim.errors.EstimatorFailure):
        mle._fit_pw(distances)

def test_on_exponential_seq_of_distances():
    np.random.seed(356)
    SIZE = 5
    dist_matrix = __generate_distance_matrix(SIZE, 0)
    for i in range(1, SIZE):
        dist_matrix[0, i] = np.exp(i)
        dist_matrix[i, 0] = np.exp(i)

    mle = skdim.id_flex.MLE_basic(metric="precomputed", nbhd_type = 'knn', n_neighbors=2)
    estim_dim = mle.fit_transform_pw(dist_matrix)[0]
    assert 1.0 == pytest.approx(estim_dim)

    mle = skdim.id_flex.MLE_basic(metric="precomputed", nbhd_type = 'knn', n_neighbors=3)
    estim_dim = mle.fit_transform_pw(dist_matrix)[0]
    assert 2.0 / 3.0 == pytest.approx(estim_dim)

    mle = skdim.id_flex.MLE_basic(metric="precomputed", nbhd_type = 'knn', n_neighbors=4)
    estim_dim = mle.fit_transform_pw(dist_matrix)[0]
    assert 0.5  == pytest.approx(estim_dim)

def test_exception_is_raised_when_neighbourhoods_empty():
    np.random.seed(123)
    dist_matrix = __generate_distance_matrix(10, 12)
    mle = skdim.id_flex.MLE_basic(nbhd_type = 'eps', metric = 'precomputed', radius = 5)
    with pytest.raises(ValueError): 
        mle.fit_transform(dist_matrix)

def test_when_eps_and_knn_almost_equivalent():
    
    rectangle = np.zeros((4,4))
    rectangle[1:3, 1] = 2
    rectangle[:2, 0] = 1

    knn_mle = skdim.id_flex.MLE_basic(nbhd_type = 'knn', n_neighbors=2)
    eps_mle = skdim.id_flex.MLE_basic(nbhd_type = 'eps', radius = 2)

    knn_estim = knn_mle.fit_transform(rectangle)
    eps_estim = eps_mle.fit_transform(rectangle)

    assert knn_estim * 2.0 == pytest.approx(eps_estim)

def test_estim_decrease_when_eps_bigger_then_set_diameter():
    np.random.seed(567)
    eps_list = [10, 20, 30, 40, 50, 60]
    dist_matrix = __generate_distance_matrix(10, 0, 10)
    results = []
    for eps in eps_list:
        mle = skdim.id_flex.MLE_basic( nbhd_type = 'eps', metric="precomputed", radius=eps)
        results.append(mle.fit_transform(dist_matrix))
    for i in range(1, len(results)):
        assert results[i] < results[i-1]