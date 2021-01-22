# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

jax = pytest.importorskip("jax")
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


def test_jax_interop_1():
    j = jax.numpy.arange(10)
    n = np.arange(10, dtype=np.int32)
    jax_index_arr = ak.layout.Index64.from_jax(j)
    np_index_arr = ak.layout.Index64(n)

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
