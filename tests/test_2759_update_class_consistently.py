# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak


class ArrayBehavior(ak.Array):
    def impl(self):
        return True


class RecordBehavior(ak.Record):
    def impl(self):
        return True


BEHAVIOR = {("*", "impl"): ArrayBehavior, "impl": RecordBehavior}


def test_array_layout():
    array = ak.Array([{"x": 1}, {"y": 3}], behavior=BEHAVIOR)
    assert not isinstance(array, ArrayBehavior)

    array.layout = ak.with_name([{"x": 1}, {"y": 3}], "impl", highlevel=False)
    assert isinstance(array, ArrayBehavior)
    assert array.impl()


def test_array_behavior():
    array = ak.Array([{"x": 1}, {"y": 3}], with_name="impl")
    assert not isinstance(array, ArrayBehavior)

    array.behavior = BEHAVIOR
    assert isinstance(array, ArrayBehavior)
    assert array.impl()


def test_record_layout():
    record = ak.Record({"x": 1}, behavior=BEHAVIOR)
    assert not isinstance(record, RecordBehavior)

    record.layout = ak.with_name({"x": 1}, "impl", highlevel=False)
    assert isinstance(record, RecordBehavior)
    assert record.impl()


def test_record_behavior():
    record = ak.Record({"x": 1}, with_name="impl")
    assert not isinstance(record, RecordBehavior)

    record.behavior = BEHAVIOR
    assert isinstance(record, RecordBehavior)
    assert record.impl()
