# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import hypothesis_awkward.strategies as st_ak
import numpy as np
from hypothesis import given

import awkward as ak


@given(
    a=st_ak.constructors.arrays(
        allow_empty=False,
        allow_string=False,
        allow_bytestring=False,
        allow_list=False,
        allow_list_offset=False,
        allow_record=False,
        allow_union=False,
        allow_indexed_option=False,
        allow_byte_masked=False,
        allow_bit_masked=False,
        allow_unmasked=False,
    )
)
def test_roundtrip(a: ak.Array) -> None:
    """`to_numpy` followed by `from_numpy` reconstructs the array."""
    n = ak.to_numpy(a)
    returned = ak.from_numpy(n)
    assert ak.array_equal(a, returned, equal_nan=True)


@given(
    a=st_ak.constructors.arrays(
        # to_numpy(allow_missing=True) raises on timedelta64 with missing
        # values (create_missing_data calls np.iinfo on the "m8" dtype)
        dtypes=st_ak.supported_dtypes().filter(lambda d: d.kind != "m"),
        # from_numpy crashes on multidimensional MaskedArrays that to_numpy
        # builds from option-of-regular layouts: with no mask and a string
        # dtype it calls `.size` on a `ListArray`; with a mask, ndim >= 3,
        # and a zero-length dimension it attempts a `reshape(-1, 0)`
        allow_regular=False,
        allow_list=False,
        allow_list_offset=False,
        allow_record=False,
        allow_union=False,
    )
)
def test_roundtrip_stable(a: ak.Array) -> None:
    """`from_numpy` then `to_numpy` reproduces `to_numpy`'s output.

    `to_numpy` can lose type information (an unknown type becomes
    `float64`, trailing NULs are stripped from strings, tuples become named
    records), so `from_numpy` cannot always reconstruct the original array,
    but the converted form is a fixed point: converting the reconstruction
    yields the same NumPy array.
    """
    n = ak.to_numpy(a)
    r = ak.from_numpy(n)
    m = ak.to_numpy(r)
    # itemsize of "U"/"S" dtypes is not stable: trailing NULs widen `n`'s
    # dtype but are stripped in the roundtrip, narrowing `m`'s
    np.testing.assert_array_equal(n, m, strict=(n.dtype.kind not in "SU"))
    # assert_array_equal treats masked positions as equal to anything (even
    # under differing masks), so the masks must be compared separately
    np.testing.assert_array_equal(np.ma.getmaskarray(n), np.ma.getmaskarray(m))


@given(
    a=st_ak.constructors.arrays(
        # to_numpy(allow_missing=True) raises on timedelta64 with missing
        # values (create_missing_data calls np.iinfo on the "m8" dtype)
        dtypes=st_ak.supported_dtypes().filter(lambda d: d.kind != "m"),
        allow_empty=False,
        allow_string=False,
        allow_bytestring=False,
        allow_regular=False,
        allow_list=False,
        allow_list_offset=False,
        allow_record=False,
        allow_union=False,
    )
)
def test_roundtrip_masked(a: ak.Array) -> None:
    """Option-type arrays roundtrip through `np.ma.MaskedArray`.

    `from_numpy` normalizes option layouts (e.g., `BitMaskedArray` returns
    as `ByteMaskedArray`, an all-valid mask as `UnmaskedArray`), so content
    classes are not compared.
    """
    n = ak.to_numpy(a, allow_missing=True)
    returned = ak.from_numpy(n)
    assert ak.array_equal(a, returned, equal_nan=True, same_content_types=False)
