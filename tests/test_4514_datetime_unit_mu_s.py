# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import pytest

import awkward as ak
from awkward.types import ListType, NumpyType
from awkward.types._awkward_datashape_parser import UnexpectedInput

# "\u03bc" is the Greek small letter mu, which NumPy accepts in "\u03bcs".
# "\u00b5" is the micro sign, which NumPy does not accept.
MU_S = [
    ("datetime64[\u03bcs]", "datetime64[us]"),
    ("datetime64[3\u03bcs]", "datetime64[3us]"),
    ("timedelta64[\u03bcs]", "timedelta64[us]"),
    ("timedelta64[3\u03bcs]", "timedelta64[3us]"),
]

NOT_UNITS = [
    "datetime64[\u03bc]",
    "timedelta64[\u03bc]",
    "datetime64[\u00b5s]",
    "timedelta64[\u00b5s]",
]


@pytest.mark.parametrize(("primitive", "expected"), MU_S)
def test_numpytype_reads_mu_s(primitive, expected):
    assert str(NumpyType(primitive)) == expected


@pytest.mark.parametrize(("primitive", "expected"), MU_S)
def test_from_datashape_reads_mu_s(primitive, expected):
    assert ak.types.from_datashape(primitive, highlevel=False) == NumpyType(expected)
    assert ak.types.from_datashape(f"var * {primitive}", highlevel=False) == ListType(
        NumpyType(expected)
    )


@pytest.mark.parametrize("primitive", NOT_UNITS)
def test_numpytype_rejects_other_mu_units(primitive):
    with pytest.raises(TypeError, match="unrecognized primitive"):
        NumpyType(primitive)


@pytest.mark.parametrize("primitive", NOT_UNITS)
def test_from_datashape_rejects_other_mu_units(primitive):
    with pytest.raises(UnexpectedInput):
        ak.types.from_datashape(primitive, highlevel=False)
