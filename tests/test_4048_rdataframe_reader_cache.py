# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest

import awkward as ak

pytest.importorskip("ROOT")


@pytest.mark.parametrize("flatlist_as_rvec", [True, False])
def test_consecutive_data_sources_with_different_schemas(flatlist_as_rvec):
    # Reuse one column name with a different type, then change the column name.
    # All three sources must coexist without reusing another schema's C++ class.
    schemas = [
        {"cache_value": ak.Array([1.5, 2.5, 3.5])},
        {"cache_value": ak.Array([[1, 2], [], [3]])},
        {"cache_other": ak.Array([10, 20, 30])},
    ]
    frames = [
        ak.to_rdataframe(schema, flatlist_as_rvec=flatlist_as_rvec)
        for schema in schemas
    ]
    assert frames[0].GetColumnType("cache_value") == "double"
    assert frames[1].GetColumnType("cache_value") != "double"
    for frame, schema in zip(frames, schemas, strict=True):
        assert set(frame.GetColumnNames()) == set(schema)
        name, array = next(iter(schema.items()))
        column_type = frame.GetColumnType(name)
        result = list(frame.Take[column_type](name).GetValue())
        expected = array.to_list()
        if isinstance(expected[0], list):
            result = [list(row) for row in result]
        # ROOT implicit multithreading may already be enabled by another test.
        assert sorted(result) == sorted(expected)
