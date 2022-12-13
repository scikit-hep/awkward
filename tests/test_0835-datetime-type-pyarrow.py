# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import datetime

import numpy as np
import pytest

import awkward as ak

pyarrow = pytest.importorskip("pyarrow")

to_list = ak.operations.to_list


def test_from_arrow():
    import awkward._connect.pyarrow

    array = awkward._connect.pyarrow.handle_arrow(
        pyarrow.array(
            [datetime.datetime(2002, 1, 23), datetime.datetime(2019, 2, 20)],
            type=pyarrow.date64(),
        )
    )
    assert to_list(array) == [
        np.datetime64("2002-01-23T00:00:00.000"),
        np.datetime64("2019-02-20T00:00:00.000"),
    ]

    array = awkward._connect.pyarrow.handle_arrow(
        pyarrow.array(
            [datetime.datetime(2002, 1, 23), datetime.datetime(2019, 2, 20)],
            type=pyarrow.date32(),
        )
    )
    assert to_list(array) == [
        np.datetime64("2002-01-23"),
        np.datetime64("2019-02-20"),
    ]

    array = awkward._connect.pyarrow.handle_arrow(
        pyarrow.array(
            [datetime.time(1, 0, 0), datetime.time(2, 30, 0)],
            type=pyarrow.time64("us"),
        )
    )
    assert to_list(array) == [
        np.datetime64("1970-01-01T01:00:00.000"),
        np.datetime64("1970-01-01T02:30:00.000"),
    ]

    array = awkward._connect.pyarrow.handle_arrow(
        pyarrow.array(
            [datetime.time(1, 0, 0), datetime.time(2, 30, 0)],
            type=pyarrow.time64("ns"),
        )
    )
    assert to_list(array) == [3600000000000, 9000000000000]

    array = awkward._connect.pyarrow.handle_arrow(
        pyarrow.array(
            [datetime.time(1, 0, 0), datetime.time(2, 30, 0)],
            type=pyarrow.time32("s"),
        )
    )
    assert to_list(array) == [
        np.datetime64("1970-01-01T01:00:00.000"),
        np.datetime64("1970-01-01T02:30:00.000"),
    ]

    array = awkward._connect.pyarrow.handle_arrow(
        pyarrow.array(
            [datetime.time(1, 0, 0), datetime.time(2, 30, 0)],
            type=pyarrow.time32("ms"),
        )
    )
    assert to_list(array) == [
        np.datetime64("1970-01-01T01:00:00.000"),
        np.datetime64("1970-01-01T02:30:00.000"),
    ]

    array = awkward._connect.pyarrow.handle_arrow(
        pyarrow.array(
            [datetime.datetime(2002, 1, 23), datetime.datetime(2019, 2, 20)],
            type=pyarrow.timestamp("s"),
        )
    )
    assert to_list(array) == [
        np.datetime64("2002-01-23T00:00:00.000"),
        np.datetime64("2019-02-20T00:00:00.000"),
    ]

    array = awkward._connect.pyarrow.handle_arrow(
        pyarrow.array(
            [datetime.datetime(2002, 1, 23), datetime.datetime(2019, 2, 20)],
            type=pyarrow.timestamp("ms"),
        )
    )
    assert to_list(array) == [
        np.datetime64("2002-01-23T00:00:00.000"),
        np.datetime64("2019-02-20T00:00:00.000"),
    ]

    array = awkward._connect.pyarrow.handle_arrow(
        pyarrow.array(
            [datetime.datetime(2002, 1, 23), datetime.datetime(2019, 2, 20)],
            type=pyarrow.timestamp("us"),
        )
    )
    assert to_list(array) == [
        np.datetime64("2002-01-23T00:00:00.000"),
        np.datetime64("2019-02-20T00:00:00.000"),
    ]

    array = awkward._connect.pyarrow.handle_arrow(
        pyarrow.array(
            [datetime.datetime(2002, 1, 23), datetime.datetime(2019, 2, 20)],
            type=pyarrow.timestamp("ns"),
        )
    )
    assert to_list(array) == [1011744000000000000, 1550620800000000000]

    array = awkward._connect.pyarrow.handle_arrow(
        pyarrow.array(
            [datetime.timedelta(5), datetime.timedelta(10)],
            type=pyarrow.duration("s"),
        )
    )
    assert to_list(array) == [
        np.timedelta64(5, "D"),
        np.timedelta64(10, "D"),
    ]

    array = awkward._connect.pyarrow.handle_arrow(
        pyarrow.array(
            [datetime.timedelta(5), datetime.timedelta(10)],
            type=pyarrow.duration("ms"),
        )
    )
    assert to_list(array) == [
        np.timedelta64(5, "D"),
        np.timedelta64(10, "D"),
    ]

    array = awkward._connect.pyarrow.handle_arrow(
        pyarrow.array(
            [datetime.timedelta(5), datetime.timedelta(10)],
            type=pyarrow.duration("us"),
        )
    )
    assert to_list(array) == [
        np.timedelta64(5, "D"),
        np.timedelta64(10, "D"),
    ]

    array = awkward._connect.pyarrow.handle_arrow(
        pyarrow.array(
            [datetime.timedelta(5), datetime.timedelta(10)],
            type=pyarrow.duration("ns"),
        )
    )
    assert to_list(array) == [432000000000000, 864000000000000]
