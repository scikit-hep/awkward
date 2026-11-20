# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import numpy as np
import pytest

import awkward as ak

numba = pytest.importorskip("numba")

ak.numba.register_and_check()


def test_numba_ufunc_nep_50():
    @numba.vectorize(nopython=True)
    def add(x, y):
        return x + y

    array = ak.values_astype([[1, 2, 3], [4]], np.int8)

    # FIXME: what error will Numba throw here for an out-of-bounds integer?
    with pytest.warns(FutureWarning, match=r"not create a writeable array"):
        result = add(array, np.int16(np.iinfo(np.int8).max + 1))

    flattened = ak.to_numpy(ak.flatten(result))
    assert flattened.dtype == np.dtype(np.int64)
