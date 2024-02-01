# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import packaging.version
import pytest

import awkward as ak

numba = pytest.importorskip("numba")

NUMBA_HAS_NEP_50 = packaging.version.parse(
    numba.__version__
) >= packaging.version.Version("0.59.0")
NUMBA_OLDER_THAN_58 = packaging.version.parse(
    numba.__version__
) < packaging.version.Version("0.58.0")

ak.numba.register_and_check()


@pytest.mark.skipif(not NUMBA_HAS_NEP_50, reason="Numba does not have NEP-50 support")
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


@pytest.mark.skipif(NUMBA_HAS_NEP_50, reason="Numba has NEP-50 support")
@pytest.mark.skipif(
    NUMBA_OLDER_THAN_58, reason="Numba has a known bug with type dispatch for <0.58"
)
def test_numba_ufunc_legacy():
    @numba.vectorize(nopython=True)
    def add(x, y):
        return x + y

    array = ak.values_astype([[1, 2, 3], [4]], np.int8)
    with pytest.warns(FutureWarning, match=r"not create a writeable array"):
        result = add(array, np.int16(np.iinfo(np.int8).max + 1))

    flattened = ak.to_numpy(ak.flatten(result))
    # Seems like Numba chooses int64 here unless a 32-bit platform
    assert flattened.dtype == np.dtype(np.int32 if ak._util.bits32 else np.int64)
