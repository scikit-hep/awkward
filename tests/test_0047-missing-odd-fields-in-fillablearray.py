# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    out = ak.ArrayBuilder()

    out.begin_record()
    if True:
        out.field("x")
        out.integer(3)
        out.field("extreme")
        out.begin_record()
        if True:
            out.field("pt")
            out.real(3.3)
            out.field("charge")
            out.integer(-1)
            out.field("iso")
            out.integer(100)
        out.end_record()
    out.end_record()

    out.begin_record()
    if True:
        out.field("x")
        out.integer(3)
    out.end_record()

    ss = out.snapshot()
    assert ak.to_list(ss) == [
        {"x": 3, "extreme": {"pt": 3.3, "charge": -1, "iso": 100}},
        {"x": 3, "extreme": None},
    ]

    assert ak.to_list(
        ak.Array(
            [{"x": 3, "extreme": {"pt": 3.3, "charge": -1, "iso": 100}}, {"x": 3}],
            check_valid=True,
        )
    ) == [
        {"x": 3, "extreme": {"pt": 3.3, "charge": -1, "iso": 100}},
        {"x": 3, "extreme": None},
    ]

    assert ak.to_list(
        ak.Array(
            [
                {"x": 3, "extreme": {"pt": 3.3, "charge": -1, "iso": 100}},
                {"x": 3, "what": 3},
            ],
            check_valid=True,
        )
    ) == [
        {"x": 3, "extreme": {"pt": 3.3, "charge": -1, "iso": 100}, "what": None},
        {"x": 3, "extreme": None, "what": 3},
    ]
