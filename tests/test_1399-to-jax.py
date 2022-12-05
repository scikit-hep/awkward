# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

ak.jax.register_and_check()


def test_from_jax_1():
    ak_array_1d = ak.Array(np.arange(10))
    ak_array_2d = ak.Array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6], [7.7, 8.8]])

    jax_array_1d = ak.to_jax(ak_array_1d)
    jax_array_2d = ak.to_jax(ak_array_2d)

    for i in range(10):
        assert jax_array_1d[i] == ak_array_1d[i]

    for i in range(4):
        for j in range(2):
            assert jax_array_2d[i][j] == ak_array_2d[i][j]


def test_from_jax_2():
    content0 = ak.Array(np.array([1, 2, 3], dtype=np.int64)).layout
    content1 = ak.contents.numpyarray.NumpyArray(
        np.array([1, 2, 3, 4, 5], dtype=np.int32)
    )
    tags = ak.index.Index8(np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=np.int8))
    index = ak.index.Index32(np.array([0, 1, 0, 1, 2, 2, 4, 3], dtype=np.int32))
    unionarray = ak.contents.unionarray.UnionArray.simplified(
        tags, index, [content0, content1]
    )

    jax_array = ak.to_jax(unionarray)

    assert jax_array.tolist() == ak.to_list(unionarray)
