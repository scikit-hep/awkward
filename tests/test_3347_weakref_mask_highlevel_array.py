from __future__ import annotations

import platform

import pytest

import awkward as ak


@pytest.mark.skipif(
    platform.python_implementation() == "PyPy",
    reason="PyPy has a different GC strategy than CPython and thus weakrefs may stay alive a little bit longer than expected, see: https://doc.pypy.org/en/latest/cpython_differences.html#differences-related-to-garbage-collection-strategies",
)
def test_Array_mask_weakref():
    arr = ak.Array([1])
    m = arr.mask

    assert ak.all(m[[True]] == arr)

    del arr
    with pytest.raises(
        ValueError,
        match="The array to mask was deleted before it could be masked. If you want to construct this mask, you must either keep the array alive or use 'ak.mask' explicitly.",
    ):
        _ = m[[True]]
