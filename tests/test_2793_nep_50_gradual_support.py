# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import contextlib

import numpy as np
import pytest

import awkward as ak


@pytest.mark.parametrize("backend", ["cpu", "typetracer"])
def test_with_nep_50(backend):
    array = ak.to_backend(np.arange(255, dtype=np.uint8), backend)
    assert array.layout.dtype == np.dtype(np.uint8)

    typed_scalar = np.uint64(0)
    assert (array + typed_scalar).layout.dtype == np.dtype(np.uint64)

    # With NEP-50, we can ask NumPy to use value-less type resolution
    with pytest.raises(OverflowError) if backend == "cpu" else contextlib.nullcontext():
        untyped_scalar = 512
        assert (array + untyped_scalar).layout.dtype == np.dtype(np.uint8)
