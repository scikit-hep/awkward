# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

array = ak.Array([{"a": 1, "b": 1}, {"a": 2, "b": 2}])


def test_numpy_string_dtype():
    if not hasattr(np.dtypes, "StringDType"):
        pytest.skip("NumPy has no StringDType")
    with pytest.raises(TypeError, match=r"cannot slice with an array of dtype"):
        array[np.array(["a"], dtype=np.dtypes.StringDType())]


def test_numpy_object_dtype():
    with pytest.raises(TypeError, match=r"cannot slice with an array of dtype"):
        array[np.array(["a"], dtype=object)]


def test_field_names_still_work():
    assert array[["a"]].to_list() == [{"a": 1}, {"a": 2}]
    assert array[np.array(["a"])].to_list() == [{"a": 1}, {"a": 2}]


@pytest.mark.parametrize(
    "make_where",
    [
        lambda pd: pd.Index(["b"], dtype="string"),
        lambda pd: pd.Series(["b"], dtype="string"),
        lambda pd: pd.array([0], dtype="Int64"),
    ],
    ids=["Index-string", "Series-string", "array-Int64"],
)
def test_pandas_extension_dtype(make_where):
    pd = pytest.importorskip("pandas")
    with pytest.raises(TypeError, match=r"is not a NumPy dtype"):
        array[make_where(pd)]


def test_pandas_object_index_still_works():
    pd = pytest.importorskip("pandas")
    assert array[pd.Index(["b"], dtype=object)].to_list() == [{"b": 1}, {"b": 2}]


nested = ak.Array([[1, 2, 3], [4, 5, 6]])


def test_numpy_string_dtype_in_tuple():
    if not hasattr(np.dtypes, "StringDType"):
        pytest.skip("NumPy has no StringDType")
    with pytest.raises(TypeError, match=r"cannot slice with an array of dtype"):
        nested[:, np.array(["a"], dtype=np.dtypes.StringDType())]


@pytest.mark.parametrize(
    "make_where",
    [
        lambda pd: (slice(None), pd.array([0, 1], dtype="Int64")),
        lambda pd: (pd.array([0, 1], dtype="Int64"),),
    ],
    ids=["second-axis", "one-item-tuple"],
)
def test_pandas_extension_dtype_in_tuple(make_where):
    pd = pytest.importorskip("pandas")
    with pytest.raises(TypeError, match=r"is not a NumPy dtype"):
        nested[make_where(pd)]


def test_pandas_extension_dtype_fields_in_tuple():
    pd = pytest.importorskip("pandas")
    with pytest.raises(TypeError, match=r"is not a NumPy dtype"):
        array[(pd.Index(["b"], dtype="string"),)]


def test_supported_indexes_in_tuple_still_work():
    assert nested[:, np.array([0, 1])].to_list() == [[1, 2], [4, 5]]
    assert nested[:, [0, 1]].to_list() == [[1, 2], [4, 5]]
    assert array[(["b"],)].to_list() == [{"b": 1}, {"b": 2}]
