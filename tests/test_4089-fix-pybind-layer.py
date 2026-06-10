# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""
Tests for fixes in awkward-cpp pybind11 bindings (PR #4089):
- datetime.time conversion via ak.from_iter (was broken by _s vs _a literal bug)
- datetime.timedelta builder dispatch (copy-paste fix)
- from_iter sanity: complex, str, bytes, dict, list still work
"""

from __future__ import annotations

import datetime

import numpy as np

import awkward as ak


def test_from_iter_time_basic():
    """datetime.time objects round-trip through ak.from_iter."""
    t = datetime.time(1, 2, 3)
    arr = ak.from_iter([t])
    assert arr.type == ak.types.ArrayType(ak.types.NumpyType("datetime64[us]"), 1)
    # Should produce a datetime64[us] at 1970-01-01T01:02:03
    assert str(arr[0]) == "1970-01-01T01:02:03.000000"


def test_from_iter_time_midnight():
    """datetime.time(0, 0, 0) maps to midnight on epoch date."""
    arr = ak.from_iter([datetime.time(0, 0, 0)])
    assert str(arr[0]) == "1970-01-01T00:00:00.000000"


def test_from_iter_time_multiple():
    """Multiple datetime.time values are stored correctly."""
    times = [
        datetime.time(0, 0, 0),
        datetime.time(12, 30, 45),
        datetime.time(23, 59, 59),
    ]
    arr = ak.from_iter(times)
    assert len(arr) == 3
    expected_strs = [
        "1970-01-01T00:00:00.000000",
        "1970-01-01T12:30:45.000000",
        "1970-01-01T23:59:59.000000",
    ]
    for i, exp in enumerate(expected_strs):
        assert str(arr[i]) == exp


def test_from_iter_timedelta_datetime():
    """datetime.timedelta objects use the timedelta builder (not datetime)."""
    td = datetime.timedelta(hours=1, minutes=30)
    arr = ak.from_iter([td])
    assert arr.type == ak.types.ArrayType(ak.types.NumpyType("timedelta64[us]"), 1)
    # 1h 30min = 5400 seconds = 5400000000 microseconds
    expected_us = (1 * 3600 + 30 * 60) * 1_000_000
    val = arr[0]
    assert int(val.view(np.int64)) == expected_us


def test_from_iter_complex():
    """Complex numbers still work after replacing builder_fromiter_iscomplex."""
    arr = ak.from_iter([1 + 2j, 3 + 4j])
    assert len(arr) == 2
    assert arr[0].item() == 1 + 2j
    assert arr[1].item() == 3 + 4j


def test_from_iter_strings():
    """String iteration still works."""
    arr = ak.from_iter(["hello", "world"])
    assert list(arr) == ["hello", "world"]


def test_from_iter_bytes():
    """Bytes iteration still works."""
    arr = ak.from_iter([b"hello", b"world"])
    assert list(arr) == [b"hello", b"world"]


def test_from_iter_dict():
    """Dict (record) iteration still works."""
    arr = ak.from_iter([{"x": 1, "y": 2}, {"x": 3, "y": 4}])
    assert list(arr["x"]) == [1, 3]
    assert list(arr["y"]) == [2, 4]


def test_from_iter_nested_list():
    """Nested list iteration still works."""
    arr = ak.from_iter([[1, 2, 3], [4, 5]])
    assert len(arr) == 2
    assert list(arr[0]) == [1, 2, 3]
    assert list(arr[1]) == [4, 5]


def test_from_iter_numpy_datetime64():
    """numpy.datetime64 still works."""
    arr = ak.from_iter([np.datetime64("2020-01-01")])
    assert len(arr) == 1


def test_from_iter_numpy_timedelta64():
    """numpy.timedelta64 still works."""
    arr = ak.from_iter([np.timedelta64(100, "us")])
    assert len(arr) == 1
    assert int(arr[0].view(np.int64)) == 100


def test_from_iter_datetime_datetime():
    """datetime.datetime still works."""
    dt = datetime.datetime(2020, 6, 15, 10, 30, 0)
    arr = ak.from_iter([dt])
    assert len(arr) == 1


def test_from_iter_datetime_date():
    """datetime.date still works."""
    d = datetime.date(2020, 6, 15)
    arr = ak.from_iter([d])
    assert len(arr) == 1
