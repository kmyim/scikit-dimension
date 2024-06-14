import pytest
import numpy as np
import skdim.id_flex
from sklearn.datasets import make_swiss_roll
import skdim.errors

def test_on_swiss_roll():
    np.random.seed(782)
    swiss_roll_dat = make_swiss_roll(1000)
    cdim = skdim.id_flex.CDim()
    estim_dim = cdim.fit_transform(swiss_roll_dat[0], nbhd_type = 'knn', n_neighbors=10)
    # The expected value is taken from Yang et al
    assert 2.0 == pytest.approx(estim_dim, 0.02)
