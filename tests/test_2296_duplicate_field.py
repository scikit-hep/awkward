# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import re

import pytest

import awkward as ak


@pytest.mark.parametrize("forget_length", [False, True])
def test_invalid(forget_length):
    layout = ak.contents.RecordArray(
        [
            ak.contents.NumpyArray([1, 2, 3]),
            ak.contents.NumpyArray([1, 2, 3]),
        ],
        ["x", "x"],
    )
    assert re.match(r".*duplicate field 'x'.*", ak.validity_error(layout)) is not None
    assert (
        re.match(
            r".*duplicate field 'x'.*",
            ak.validity_error(layout.to_typetracer(forget_length)),
        )
        is not None
    )


@pytest.mark.parametrize("forget_length", [False, True])
def test_valid(forget_length):
    layout = ak.contents.RecordArray(
        [
            ak.contents.NumpyArray([1, 2, 3]),
            ak.contents.NumpyArray([1, 2, 3]),
        ],
        ["x", "y"],
    )
    assert re.match(r".*duplicate field 'x'.*", ak.validity_error(layout)) is None
    assert (
        re.match(
            r".*duplicate field 'x'.*",
            ak.validity_error(layout.to_typetracer(forget_length)),
        )
        is None
    )
