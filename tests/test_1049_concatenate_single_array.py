# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak


def test_single_numpy_array():
    array = np.arange(4 * 3 * 2).reshape(4, 3, 2)
    result = ak.to_numpy(ak.concatenate(array))
    assert result.tolist() == [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9],
        [10, 11],
        [12, 13],
        [14, 15],
        [16, 17],
        [18, 19],
        [20, 21],
        [22, 23],
    ]


def test_single_awkward_array():
    array = ak.from_iter([[1, 2, 3], [4, 5, 6, 7], [8, 9]])
    result = ak.concatenate(array)
    assert result.to_list() == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_single_jax_array():
    jnp = pytest.importorskip("jax.numpy")
    ak.jax.register_and_check()

    array = jnp.arange(4 * 3 * 2).reshape(4, 3, 2)
    result = ak.concatenate(array)
    assert result.to_list() == [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9],
        [10, 11],
        [12, 13],
        [14, 15],
        [16, 17],
        [18, 19],
        [20, 21],
        [22, 23],
    ]
