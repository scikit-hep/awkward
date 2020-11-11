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
    assert awkward1.to_pandas(tuples)["1"].values.tolist() == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5]

    offsets = awkward1.layout.Index64(numpy.array([0, 1, 1, 3, 4], dtype=numpy.int64))
    nested = awkward1.Array(awkward1.layout.ListOffsetArray64(offsets, indexed.layout))
    assert awkward1.to_pandas(nested)["values"].values.tolist() == [3.3, 3.3, 1.1, 5.5]

    offsets2 = awkward1.layout.Index64(numpy.array([0, 3, 3, 4, 6], dtype=numpy.int64))
    nested2 = awkward1.Array(awkward1.layout.ListOffsetArray64(offsets2, tuples.layout))

    assert awkward1.to_pandas(nested2)["1"].values.tolist() == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5]

    recrec = awkward1.Array([{"x": {"y": 1}}, {"x": {"y": 2}}, {"x": {"y": 3}}])
    assert awkward1.to_pandas(recrec)["x", "y"].values.tolist() == [1, 2, 3]

    recrec2 = awkward1.Array([{"x": {"a": 1, "b": 2}, "y": {"c": 3, "d": 4}}, {"x": {"a": 10, "b": 20}, "y": {"c": 30, "d": 40}}])
    assert awkward1.to_pandas(recrec2)["y", "c"].values.tolist() == [3, 30]

    recrec3 = awkward1.Array([{"x": 1, "y": {"c": 3, "d": 4}}, {"x": 10, "y": {"c": 30, "d": 40}}])
    assert awkward1.to_pandas(recrec3)["y", "c"].values.tolist() == [3, 30]

    tuptup = awkward1.Array([(1.0, (1.1, 1.2)), (2.0, (2.1, 2.2)), (3.0, (3.1, 3.2))])
    assert awkward1.to_pandas(tuptup)["1", "0"].values.tolist() == [1.1, 2.1, 3.1]

    recrec4 = awkward1.Array([[{"x": 1, "y": {"c": 3, "d": 4}}], [{"x": 10, "y": {"c": 30, "d": 40}}]])
    assert awkward1.to_pandas(recrec4)["y", "c"].values.tolist() == [3, 30]

def test_broken():
    ex = awkward1.Array([[1, 2, 3], [], [4, 5]])
    p4 = awkward1.zip({"x": ex})
    p4c = awkward1.cartesian({"a": p4, "b": p4})
    df = awkward1.to_pandas(p4c)
    assert df["a", "x"].values.tolist() == [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5]
    assert df["b", "x"].values.tolist() == [1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 4, 5]

def test_union_to_record():
    recordarray1 = awkward1.Array([{"x": 1, "y": 1.1}, {"x": 3, "y": 3.3}]).layout
    recordarray2 = awkward1.Array([{"y": 2.2, "z": 999}]).layout
    tags = awkward1.layout.Index8(numpy.array([0, 1, 0], dtype=numpy.int8))
    index = awkward1.layout.Index64(numpy.array([0, 0, 1], dtype=numpy.int64))
    unionarray = awkward1.layout.UnionArray8_64(tags, index, [recordarray1, recordarray2])
    assert awkward1.to_list(unionarray) == [{"x": 1, "y": 1.1}, {"y": 2.2, "z": 999}, {"x": 3, "y": 3.3}]

    converted = awkward1._util.union_to_record(unionarray, "values")
    assert isinstance(converted, awkward1.layout.RecordArray)
    assert awkward1.to_list(converted) == [{"x": 1, "y": 1.1, "z": None}, {"x": None, "y": 2.2, "z": 999}, {"x": 3, "y": 3.3, "z": None}]

    otherarray = awkward1.Array(["one", "two"]).layout
    tags2 = awkward1.layout.Index8(numpy.array([0, 2, 1, 2, 0], dtype=numpy.int8))
    index2 = awkward1.layout.Index64(numpy.array([0, 0, 0, 1, 1], dtype=numpy.int64))
    unionarray2 = awkward1.layout.UnionArray8_64(tags2, index2, [recordarray1, recordarray2, otherarray])
    assert awkward1.to_list(unionarray2) == [{"x": 1, "y": 1.1}, "one", {"y": 2.2, "z": 999}, "two", {"x": 3, "y": 3.3}]

    converted2 = awkward1._util.union_to_record(unionarray2, "values")
    assert isinstance(converted2, awkward1.layout.RecordArray)
    assert awkward1.to_list(converted2) == [{"x": 1, "y": 1.1, "z": None, "values": None}, {"x": None, "y": None, "z": None, "values": "one"}, {"x": None, "y": 2.2, "z": 999, "values": None}, {"x": None, "y": None, "z": None, "values": "two"}, {"x": 3, "y": 3.3, "z": None, "values": None}]

    df_unionarray = awkward1.to_pandas(unionarray)
    numpy.testing.assert_array_equal(df_unionarray["x"].values, numpy.array([1, numpy.nan, 3]))
    numpy.testing.assert_array_equal(df_unionarray["y"].values, numpy.array([1.1, 2.2, 3.3]))
    numpy.testing.assert_array_equal(df_unionarray["z"].values, numpy.array([numpy.nan, 999, numpy.nan]))

    df_unionarray2 = awkward1.to_pandas(unionarray2)
    numpy.testing.assert_array_equal(df_unionarray2["x"].values, [1, numpy.nan, numpy.nan, numpy.nan, 3])
    numpy.testing.assert_array_equal(df_unionarray2["y"].values, [1.1, numpy.nan, 2.2, numpy.nan, 3.3])
    numpy.testing.assert_array_equal(df_unionarray2["z"].values, [numpy.nan, numpy.nan, 999, numpy.nan, numpy.nan])
    numpy.testing.assert_array_equal(df_unionarray2["values"].values, ["nan", "one", "nan", "two", "nan"])
