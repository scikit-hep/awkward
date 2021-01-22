# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

jax = pytest.importorskip("jax")
jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_enable_x64", True)


def test_from_jax_1():
    ak_array_1d = ak.to_kernels(ak.Array(np.arange(10)), "cuda")
    ak_array_2d = ak.to_kernels(
        ak.Array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6], [7.7, 8.8]]), "cuda"
    )

    jax_array_1d = ak.to_jax(ak_array_1d)
    jax_array_2d = ak.to_jax(ak_array_2d)

    for i in range(10):
        assert jax_array_1d[i] == ak_array_1d[i]

    for i in range(4):
        for j in range(2):
            assert jax_array_2d[i][j] == ak_array_2d[i][j]


@pytest.mark.skip(
    reason="merging GPU arrays not yet implemented on GPUs (NumpyArray_fill)"
)
def test_from_jax_2():
    content0 = ak.Array(np.array([1, 2, 3], dtype=np.int64)).layout
    content1 = ak.layout.NumpyArray(np.array([1, 2, 3, 4, 5], dtype=np.int32))
    tags = ak.layout.Index8(np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=np.int8))
    index = ak.layout.Index32(np.array([0, 1, 0, 1, 2, 2, 4, 3], dtype=np.int32))
    unionarray = ak.layout.UnionArray8_32(tags, index, [content0, content1])

    unionarray_cuda = unionarray.copy_to("cuda")

    ak.to_jax(unionarray_cuda)
