# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

ak.jax.register_and_check()


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


def test_NumpyArray_constructor():
    assert ak.backend(ak.contents.NumpyArray(np.array([1, 2, 3]))) == "cpu"
    assert ak.backend(ak.contents.NumpyArray(jax.numpy.array([1, 2, 3]))) == "jax"


def test_add_2():
    one = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], backend="jax")
    two = 100
    assert ak.backend(one) == "jax"
    three = one + two
    assert ak.to_list(three) == [[101.1, 102.2, 103.3], [], [104.4, 105.5]]
    assert ak.backend(three) == "jax"
