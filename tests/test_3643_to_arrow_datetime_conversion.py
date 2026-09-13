# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import datetime

import numpy as np
import pytest

import awkward as ak

pyarrow = pytest.importorskip("pyarrow")
pytest.importorskip("awkward._connect.pyarrow")

# gh-3643: ``np.datetime64[D]`` must convert to a pyarrow ``date32`` (days since
# epoch), not be mishandled by the generic ``from_numpy_dtype`` mapping.

DAYS = np.array(["2021-01-01", "2021-06-15", "1999-12-31"], dtype="datetime64[D]")
EXPECTED = [
    datetime.date(2021, 1, 1),
    datetime.date(2021, 6, 15),
    datetime.date(1999, 12, 31),
]


def test_datetime64_day_converts_to_date32():
    array = ak.Array(DAYS)
    arrow = ak.to_arrow(array, extensionarray=False)
    assert arrow.type == pyarrow.date32()
    assert arrow.to_pylist() == EXPECTED


def test_datetime64_day_arrow_roundtrip():
    array = ak.Array(DAYS)
    result = ak.from_arrow(ak.to_arrow(array))
    assert result.to_list() == array.to_list()
    assert str(result.type) == "3 * datetime64[D]"


def test_datetime64_day_inside_list_roundtrips():
    content = ak.contents.NumpyArray(DAYS)
    listarray = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 2, 3], dtype=np.int64)), content
    )
    array = ak.Array(listarray)
    result = ak.from_arrow(ak.to_arrow(array))
    assert result.to_list() == array.to_list()
