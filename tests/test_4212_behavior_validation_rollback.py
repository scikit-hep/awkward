# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import annotations

import pytest

import awkward as ak


class NoNewFieldsError(Exception):
    pass


class OnlyXY:
    def __awkward_validation__(self) -> None:
        if set(self.fields) != {"x", "y"}:
            raise NoNewFieldsError(f"got fields: {sorted(self.fields)}")


class OnlyXYArray(OnlyXY, ak.Array):
    pass


class OnlyXYRecord(OnlyXY, ak.Record):
    pass


behavior = {"xy": OnlyXYRecord, ("*", "xy"): OnlyXYArray}


def make_array():
    return ak.Array([{"x": 1.1, "y": 2.2}], with_name="xy", behavior=behavior)


def test_setitem():
    array = make_array()
    with pytest.raises(NoNewFieldsError):
        array["z"] = [3.3]
    assert array.fields == ["x", "y"]
    assert array.to_list() == [{"x": 1.1, "y": 2.2}]


def test_delitem():
    array = make_array()
    with pytest.raises(NoNewFieldsError):
        del array["y"]
    assert array.fields == ["x", "y"]


def test_behavior_setter():
    array = ak.Array([{"x": 1.1, "y": 2.2, "z": 3.3}], with_name="xy")
    with pytest.raises(NoNewFieldsError):
        array.behavior = behavior
    assert array.behavior is None
    assert not isinstance(array, OnlyXYArray)


def test_record_setitem():
    record = make_array()[0]
    with pytest.raises(NoNewFieldsError):
        record["z"] = 3.3
    assert record.fields == ["x", "y"]
    assert record.to_list() == {"x": 1.1, "y": 2.2}
