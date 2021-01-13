# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import jax.numpy as jnp  # noqa: F401
import awkward as ak  # noqa: F401


def test_jax_interop():
    j = jnp.arange(10, dtype='i4')
    print(j.__cuda_array_interface__)
    n = np.arange(10, dtype=np.int32)
    jax_index_arr = ak.layout.Index32.from_jax(j)
    np_index_arr = ak.layout.Index32(n)

    # GPU->CPU
    assert ak.to_list(np.asarray(jax_index_arr.copy_to("cpu"))) == ak.to_list(
        np.asarray(np_index_arr)
    )
    # CPU->CPU
    assert ak.to_list(np.asarray(np_index_arr.copy_to("cpu"))) == ak.to_list(
        np.asarray(np_index_arr)
    )
    # CPU->GPU->CPU
    assert ak.to_list(np.asarray(np_index_arr)) == ak.to_list(
        np.asarray(np_index_arr.copy_to("cuda").copy_to("cpu"))
    )
