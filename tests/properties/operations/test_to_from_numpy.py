# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import hypothesis_awkward.strategies as st_ak
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
