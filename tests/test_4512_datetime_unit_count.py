# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import pytest

import awkward as ak
from awkward.types import ListType, NumpyType
from awkward.types._awkward_datashape_parser import UnexpectedInput

# NumPy accepts ASCII whitespace before the count of a datetime unit.
ACCEPTED = [
    ("datetime64[ 3us]", "datetime64[3us]"),
    ("timedelta64[\t\n7ns]", "timedelta64[7ns]"),
    ("datetime64[\x0b\x0c\r3us]", "datetime64[3us]"),
    ("datetime64[007s]", "datetime64[7s]"),
    ("timedelta64[2147483647s]", "timedelta64[2147483647s]"),
]

# NumPy rejects these units.
REJECTED = [
    "datetime64[-3us]",  # negative count
    "timedelta64[-3us]",
    "datetime64[ us]",  # whitespace without a count
    "timedelta64[\t us]",
    "datetime64[\x853us]",  # non-ASCII whitespace (U+0085)
    "datetime64[\u00a03us]",  # non-ASCII whitespace (U+00A0)
    "timedelta64[\u20033us]",  # non-ASCII whitespace (U+2003)
]


@pytest.mark.parametrize(("primitive", "expected"), ACCEPTED)
def test_numpytype_reads_count(primitive, expected):
    assert str(NumpyType(primitive)) == expected


@pytest.mark.parametrize(("primitive", "expected"), ACCEPTED)
def test_from_datashape_reads_count(primitive, expected):
    assert ak.types.from_datashape(primitive, highlevel=False) == NumpyType(expected)
    assert ak.types.from_datashape(f"var * {primitive}", highlevel=False) == ListType(
        NumpyType(expected)
    )


@pytest.mark.parametrize("primitive", REJECTED)
def test_numpytype_rejects_count(primitive):
    with pytest.raises(TypeError, match="unrecognized primitive"):
        NumpyType(primitive)


@pytest.mark.parametrize("primitive", REJECTED)
def test_from_datashape_rejects_count(primitive):
    with pytest.raises(UnexpectedInput):
        ak.types.from_datashape(primitive, highlevel=False)
    with pytest.raises(UnexpectedInput):
        ak.types.from_datashape(f"var * {primitive}", highlevel=False)
