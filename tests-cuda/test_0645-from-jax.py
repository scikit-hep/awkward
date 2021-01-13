# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import jax.numpy as jnp  # noqa: F401
import awkward as ak  # noqa: F401


def test_from_jax():
    jax_array_1d = jnp.arange(10)
    jax_array_2d = jnp.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6], [7.7, 8.8]])

    ak_jax_array_1d = ak.from_jax(jax_array_1d)
    ak_jax_array_2d = ak.from_jax(jax_array_2d)

    for i in range(10):
        assert ak_jax_array_1d[i] == jax_array_1d[i]

    for i in range(4):
        for j in range(2):
            assert ak_jax_array_2d[i][j] == jax_array_2d[i][j]


def test_from_jax_tolist():
    jax_array_1d = jnp.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

    ak_jax_array_1d = ak.from_jax(jax_array_1d)

    assert ak.to_list(ak_jax_array_1d.layout) == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]


def test_NumpyArray_constructor():
    assert ak.kernels(ak.layout.NumpyArray(np.array([1, 2, 3]))) == "cpu"
    assert ak.kernels(ak.layout.NumpyArray(jnp.array([1, 2, 3]))) == "cuda"
