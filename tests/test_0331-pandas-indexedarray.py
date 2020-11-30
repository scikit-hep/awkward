# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest
import numpy as np
import awkward1 as ak


pandas = pytest.importorskip("pandas")


def test():
    simple = ak.Array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])
    assert ak.to_pandas(simple)["values"].values.tolist() == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5]

    index = ak.layout.Index64(np.array([3, 3, 1, 5], dtype=np.int64))
    indexed = ak.Array(ak.layout.IndexedArray64(index, simple.layout))
    assert indexed.tolist() == [3.3, 3.3, 1.1, 5.5]

    assert ak.to_pandas(indexed)["values"].values.tolist() == [3.3, 3.3, 1.1, 5.5]

    tuples = ak.Array(ak.layout.RecordArray([simple.layout, simple.layout]))
    assert ak.to_pandas(tuples)["1"].values.tolist() == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5]

    offsets = ak.layout.Index64(np.array([0, 1, 1, 3, 4], dtype=np.int64))
    nested = ak.Array(ak.layout.ListOffsetArray64(offsets, indexed.layout))
    assert ak.to_pandas(nested)["values"].values.tolist() == [3.3, 3.3, 1.1, 5.5]

    offsets2 = ak.layout.Index64(np.array([0, 3, 3, 4, 6], dtype=np.int64))
    nested2 = ak.Array(ak.layout.ListOffsetArray64(offsets2, tuples.layout))

    assert ak.to_pandas(nested2)["1"].values.tolist() == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5]

    recrec = ak.Array([{"x": {"y": 1}}, {"x": {"y": 2}}, {"x": {"y": 3}}])
    assert ak.to_pandas(recrec)["x", "y"].values.tolist() == [1, 2, 3]

    recrec2 = ak.Array([{"x": {"a": 1, "b": 2}, "y": {"c": 3, "d": 4}}, {"x": {"a": 10, "b": 20}, "y": {"c": 30, "d": 40}}])
    assert ak.to_pandas(recrec2)["y", "c"].values.tolist() == [3, 30]

    recrec3 = ak.Array([{"x": 1, "y": {"c": 3, "d": 4}}, {"x": 10, "y": {"c": 30, "d": 40}}])
    assert ak.to_pandas(recrec3)["y", "c"].values.tolist() == [3, 30]

    tuptup = ak.Array([(1.0, (1.1, 1.2)), (2.0, (2.1, 2.2)), (3.0, (3.1, 3.2))])
    assert ak.to_pandas(tuptup)["1", "0"].values.tolist() == [1.1, 2.1, 3.1]

    recrec4 = ak.Array([[{"x": 1, "y": {"c": 3, "d": 4}}], [{"x": 10, "y": {"c": 30, "d": 40}}]])
    assert ak.to_pandas(recrec4)["y", "c"].values.tolist() == [3, 30]

def test_broken():
    ex = ak.Array([[1, 2, 3], [], [4, 5]])
    p4 = ak.zip({"x": ex})
    p4c = ak.cartesian({"a": p4, "b": p4})
    df = ak.to_pandas(p4c)
    assert df["a", "x"].values.tolist() == [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5]
    assert df["b", "x"].values.tolist() == [1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 4, 5]

def test_union_to_record():
    recordarray1 = ak.Array([{"x": 1, "y": 1.1}, {"x": 3, "y": 3.3}]).layout
    recordarray2 = ak.Array([{"y": 2.2, "z": 999}]).layout
    tags = ak.layout.Index8(np.array([0, 1, 0], dtype=np.int8))
    index = ak.layout.Index64(np.array([0, 0, 1], dtype=np.int64))
    unionarray = ak.layout.UnionArray8_64(tags, index, [recordarray1, recordarray2])
    assert ak.to_list(unionarray) == [{"x": 1, "y": 1.1}, {"y": 2.2, "z": 999}, {"x": 3, "y": 3.3}]

    converted = ak._util.union_to_record(unionarray, "values")
    assert isinstance(converted, ak.layout.RecordArray)
    assert ak.to_list(converted) == [{"x": 1, "y": 1.1, "z": None}, {"x": None, "y": 2.2, "z": 999}, {"x": 3, "y": 3.3, "z": None}]

    otherarray = ak.Array(["one", "two"]).layout
    tags2 = ak.layout.Index8(np.array([0, 2, 1, 2, 0], dtype=np.int8))
    index2 = ak.layout.Index64(np.array([0, 0, 0, 1, 1], dtype=np.int64))
    unionarray2 = ak.layout.UnionArray8_64(tags2, index2, [recordarray1, recordarray2, otherarray])
    assert ak.to_list(unionarray2) == [{"x": 1, "y": 1.1}, "one", {"y": 2.2, "z": 999}, "two", {"x": 3, "y": 3.3}]

    converted2 = ak._util.union_to_record(unionarray2, "values")
    assert isinstance(converted2, ak.layout.RecordArray)
    assert ak.to_list(converted2) == [{"x": 1, "y": 1.1, "z": None, "values": None}, {"x": None, "y": None, "z": None, "values": "one"}, {"x": None, "y": 2.2, "z": 999, "values": None}, {"x": None, "y": None, "z": None, "values": "two"}, {"x": 3, "y": 3.3, "z": None, "values": None}]

    df_unionarray = ak.to_pandas(unionarray)
    np.testing.assert_array_equal(df_unionarray["x"].values, np.array([1, np.nan, 3]))
    np.testing.assert_array_equal(df_unionarray["y"].values, np.array([1.1, 2.2, 3.3]))
    np.testing.assert_array_equal(df_unionarray["z"].values, np.array([np.nan, 999, np.nan]))

    df_unionarray2 = ak.to_pandas(unionarray2)
    np.testing.assert_array_equal(df_unionarray2["x"].values, [1, np.nan, np.nan, np.nan, 3])
    np.testing.assert_array_equal(df_unionarray2["y"].values, [1.1, np.nan, 2.2, np.nan, 3.3])
    np.testing.assert_array_equal(df_unionarray2["z"].values, [np.nan, np.nan, 999, np.nan, np.nan])
    np.testing.assert_array_equal(df_unionarray2["values"].values, ["nan", "one", "nan", "two", "nan"])
