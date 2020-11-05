# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import numpy
import awkward1

pandas = pytest.importorskip("pandas")


def test():
    simple = awkward1.Array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])
    assert awkward1.to_pandas(simple)["values"].values.tolist() == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5]

    index = awkward1.layout.Index64(numpy.array([3, 3, 1, 5], dtype=numpy.int64))
    indexed = awkward1.Array(awkward1.layout.IndexedArray64(index, simple.layout))
    assert indexed.tolist() == [3.3, 3.3, 1.1, 5.5]

    assert awkward1.to_pandas(indexed)["values"].values.tolist() == [3.3, 3.3, 1.1, 5.5]

    tuples = awkward1.Array(awkward1.layout.RecordArray([simple.layout, simple.layout]))
    assert awkward1.to_pandas(tuples)["slot1"].values.tolist() == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5]

    offsets = awkward1.layout.Index64(numpy.array([0, 1, 1, 3, 4], dtype=numpy.int64))
    nested = awkward1.Array(awkward1.layout.ListOffsetArray64(offsets, indexed.layout))
    assert awkward1.to_pandas(nested)["values"].values.tolist() == [3.3, 3.3, 1.1, 5.5]

    offsets2 = awkward1.layout.Index64(numpy.array([0, 3, 3, 4, 6], dtype=numpy.int64))
    nested2 = awkward1.Array(awkward1.layout.ListOffsetArray64(offsets2, tuples.layout))

    assert awkward1.to_pandas(nested2)["slot1"].values.tolist() == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5]

    recrec = awkward1.Array([{"x": {"y": 1}}, {"x": {"y": 2}}, {"x": {"y": 3}}])
    assert awkward1.to_pandas(recrec)["x", "y"].values.tolist() == [1, 2, 3]

    recrec2 = awkward1.Array([{"x": {"a": 1, "b": 2}, "y": {"c": 3, "d": 4}}, {"x": {"a": 10, "b": 20}, "y": {"c": 30, "d": 40}}])
    assert awkward1.to_pandas(recrec2)["y", "c"].values.tolist() == [3, 30]

    recrec3 = awkward1.Array([{"x": 1, "y": {"c": 3, "d": 4}}, {"x": 10, "y": {"c": 30, "d": 40}}])
    assert awkward1.to_pandas(recrec3)["y", "c"].values.tolist() == [3, 30]

    tuptup = awkward1.Array([(1.0, (1.1, 1.2)), (2.0, (2.1, 2.2)), (3.0, (3.1, 3.2))])
    assert awkward1.to_pandas(tuptup)["slot1", "slot0"].values.tolist() == [1.1, 2.1, 3.1]

    recrec4 = awkward1.Array([[{"x": 1, "y": {"c": 3, "d": 4}}], [{"x": 10, "y": {"c": 30, "d": 40}}]])
    assert awkward1.to_pandas(recrec4)["y", "c"].values.tolist() == [3, 30]

def test_broken():
    ex = awkward1.Array([[1, 2, 3], [], [4, 5]])
    p4 = awkward1.zip({"x": ex})
    p4c = awkward1.cartesian({"a": p4, "b": p4})
    df = awkward1.to_pandas(p4c)
    assert df["a", "x"].values.tolist() == [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5]
    assert df["b", "x"].values.tolist() == [1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 4, 5]
