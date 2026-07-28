import numpy as np

from catena.eval.seed_inference import exact_sign_flip_test


def test_sign_flip_detects_consistently_positive_effect():
    values = np.ones(8)
    assert exact_sign_flip_test(values, "greater") <= 1 / 256
