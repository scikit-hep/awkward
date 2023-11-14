# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import contextlib

import numpy as np
import packaging.version
import pytest

import awkward as ak

NUMPY_HAS_NEP_50 = packaging.version.parse(np.__version__) >= packaging.version.Version(
    "1.24.0"
)


@pytest.mark.skipif(not NUMPY_HAS_NEP_50, reason="NEP-50 requires NumPy >= 1.24.0")
@pytest.mark.parametrize("backend", ["cpu", "typetracer"])
def test_with_nep_50(backend):
    array = ak.to_backend(np.arange(255, dtype=np.uint8), backend)
    assert array.layout.dtype == np.dtype(np.uint8)

    typed_scalar = np.uint64(0)
    assert (array + typed_scalar).layout.dtype == np.dtype(np.uint64)

    # With NEP-50, we can ask NumPy to use value-less type resolution
    warn_context = (
        pytest.warns(DeprecationWarning, match="out-of-bound Python integers")
        if backend == "cpu"
        else contextlib.nullcontext()
    )
    with warn_context:
        untyped_scalar = 512
        assert (array + untyped_scalar).layout.dtype == np.dtype(np.uint8)


@pytest.mark.skipif(NUMPY_HAS_NEP_50, reason="NumPy >= 1.24.0 has NEP-50 support")
@pytest.mark.parametrize("backend", ["cpu", "typetracer"])
def test_without_nep_50(backend):
    array = ak.to_backend(np.arange(255, dtype=np.uint8), backend)
    assert array.layout.dtype == np.dtype(np.uint8)

    # Without NEP-50, we still don't drop type information for typed-scalars,
    # unlike NumPy.
    typed_scalar = np.uint64(0)
    assert (array + typed_scalar).layout.dtype == np.dtype(np.uint64)

    # But, with untyped scalars, we're forced to rely on NumPy's ufunc loop resolution
    untyped_scalar = 512
    assert (array + untyped_scalar).layout.dtype == np.dtype(np.uint16)
