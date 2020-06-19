import os
import numpy as np
import random

import vegi


def r(a, b):
    return random.randint(a, b)


def random_3x3_array():
    return np.array([
        [r(1,65535), r(1,65535), r(1,65535)],
        [r(1,65535), r(1,65535), r(1,65535)],
        [r(1,65535), r(1,65535), r(1,65535)]
    ])


def test_ndvi():
    red = random_3x3_array()
    nir = random_3x3_array()

    ndvi = vegi.ndvi(red, nir)

    assert np.max(ndvi) <= 1.0
    assert np.min(ndvi) >= -1.0
