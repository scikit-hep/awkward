# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import datetime

import numpy as np

import awkward as ak


def test_from_iter_time():
    """datetime.time objects round-trip through ak.from_iter."""
    t = datetime.time(1, 2, 3)
    arr = ak.from_iter([t])
    assert arr.type == ak.types.ArrayType(ak.types.NumpyType("datetime64[us]"), 1)
    assert str(arr[0]) == "1970-01-01T01:02:03.000000"


def test_from_iter_timedelta():
    """datetime.timedelta objects use the timedelta builder (not datetime)."""
    td = datetime.timedelta(hours=1, minutes=30)
    arr = ak.from_iter([td])
    assert arr.type == ak.types.ArrayType(ak.types.NumpyType("timedelta64[us]"), 1)
    expected_us = (1 * 3600 + 30 * 60) * 1_000_000
    assert int(arr[0].view(np.int64)) == expected_us


def test_from_iter_complex():
    """Complex numbers work after replacing builder_fromiter_iscomplex with PyComplex_Check."""
    arr = ak.from_iter([1 + 2j, 3 + 4j])
    assert arr[0].item() == 1 + 2j
    assert arr[1].item() == 3 + 4j
