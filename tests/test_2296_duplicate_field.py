# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import re

import pytest  # noqa: F401

import awkward as ak


def test():
    layout = ak.contents.RecordArray(
        [
            ak.contents.NumpyArray([1, 2, 3]),
            ak.contents.NumpyArray([1, 2, 3]),
        ],
        ["x", "x"],
    )
    assert re.match(r".*duplicate field 'x'.*", ak.validity_error(layout)) is not None
