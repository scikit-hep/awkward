# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

jax = pytest.importorskip("jax")
jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_enable_x64", True)


def test_from_jax():
    jax_array_1d = jax.numpy.arange(10)
    jax_array_2d = jax.numpy.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6], [7.7, 8.8]])

    ak_jax_array_1d = ak.from_jax(jax_array_1d)
    ak_jax_array_2d = ak.from_jax(jax_array_2d)

    for i in range(10):
        assert ak_jax_array_1d[i] == jax_array_1d[i]

    for i in range(4):
        for j in range(2):
            assert ak_jax_array_2d[i][j] == jax_array_2d[i][j]


def test_from_jax_tolist():
    jax_array_1d = jax.numpy.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

    ak_jax_array_1d = ak.from_jax(jax_array_1d)

    assert ak.to_list(ak_jax_array_1d.layout) == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
