# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import json

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test_basic():
    content1 = ak.layout.NumpyArray(np.array([1, 2, 3, 4, 5], dtype=np.int64))
    content2 = ak.layout.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], dtype=np.float64)
    )
    offsets = ak.layout.Index64(np.array([0, 3, 3, 5, 6, 9], dtype=np.int64))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, content2)
    recordarray = ak.layout.RecordArray(
        [content1, listoffsetarray, content2, content1],
        keys=["one", "two", "2", "wonky"],
    )

    recordarray = v1_to_v2(recordarray)

    assert ak.to_list(recordarray.content(0)) == [1, 2, 3, 4, 5]
    assert recordarray.typetracer.content(0).form == recordarray.content(0).form
    assert ak.to_list(recordarray.content("two")) == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
        [6.6],
        [7.7, 8.8, 9.9],
    ]
    assert recordarray.typetracer.content("two").form == recordarray.content("two").form
    assert ak.to_list(recordarray.content("wonky")) == [1, 2, 3, 4, 5]
    assert (
        recordarray.typetracer.content("wonky").form
        == recordarray.content("wonky").form
    )

    str(recordarray)

    assert json.loads(ak._v2.forms.form.Form.to_json(recordarray.form)) == (
        {
            "class": "RecordArray",
            "contents": {
                "one": {
                    "class": "NumpyArray",
                    "primitive": "int64",
                    "inner_shape": [],
                    "has_identifier": False,
                    "parameters": {},
                    "form_key": None,
                },
                "two": {
                    "class": "ListOffsetArray",
                    "offsets": "i64",
                    "content": {
                        "class": "NumpyArray",
                        "primitive": "float64",
                        "inner_shape": [],
                        "has_identifier": False,
                        "parameters": {},
                        "form_key": None,
                    },
                    "has_identifier": False,
                    "parameters": {},
                    "form_key": None,
                },
                "2": {
                    "class": "NumpyArray",
                    "primitive": "float64",
                    "inner_shape": [],
                    "has_identifier": False,
                    "parameters": {},
                    "form_key": None,
                },
                "wonky": {
                    "class": "NumpyArray",
                    "primitive": "int64",
                    "inner_shape": [],
                    "has_identifier": False,
                    "parameters": {},
                    "form_key": None,
                },
            },
            "has_identifier": False,
            "parameters": {},
            "form_key": None,
        }
    )

    assert len(recordarray) == 5
    assert recordarray.index_to_field(0) == "one"
    assert recordarray.index_to_field(1) == "two"
    assert recordarray.index_to_field(2) == "2"
    assert recordarray.index_to_field(3) == "wonky"
    assert recordarray.field_to_index("wonky") == 3
    assert recordarray.field_to_index("one") == 0
    # FIXME
    # assert recordarray.field_to_index("0") == 0
    assert recordarray.field_to_index("two") == 1
    # assert recordarray.field_to_index("1") == 1
    assert recordarray.field_to_index("2") == 2
    assert recordarray.has_field("wonky")
    assert recordarray.has_field("one")
    # assert recordarray.has_field("0")
    assert recordarray.has_field("two")
    # assert recordarray.has_field("1")
    assert recordarray.has_field("2")

    assert recordarray.fields == ["one", "two", "2", "wonky"]
    assert [ak.to_list(x) for x in recordarray.contents] == [
        [1, 2, 3, 4, 5],
        [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]],
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9],
        [1, 2, 3, 4, 5],
    ]

    pairs = list(zip(recordarray.fields, recordarray.contents))
    assert pairs[0][0] == "one"
    assert pairs[1][0] == "two"
    assert pairs[2][0] == "2"
    assert pairs[3][0] == "wonky"
    assert ak.to_list(pairs[0][1]) == [1, 2, 3, 4, 5]
    assert ak.to_list(pairs[1][1]) == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
        [6.6],
        [7.7, 8.8, 9.9],
    ]
    assert ak.to_list(pairs[2][1]) == [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
    assert ak.to_list(pairs[3][1]) == [1, 2, 3, 4, 5]

    assert json.loads(ak._v2.forms.form.Form.to_json(recordarray.form)) == {
        "class": "RecordArray",
        "contents": {
            "one": {
                "class": "NumpyArray",
                "primitive": "int64",
                "inner_shape": [],
                "has_identifier": False,
                "parameters": {},
                "form_key": None,
            },
            "two": {
                "class": "ListOffsetArray",
                "offsets": "i64",
                "content": {
                    "class": "NumpyArray",
                    "primitive": "float64",
                    "inner_shape": [],
                    "has_identifier": False,
                    "parameters": {},
                    "form_key": None,
                },
                "has_identifier": False,
                "parameters": {},
                "form_key": None,
            },
            "2": {
                "class": "NumpyArray",
                "primitive": "float64",
                "inner_shape": [],
                "has_identifier": False,
                "parameters": {},
                "form_key": None,
            },
            "wonky": {
                "class": "NumpyArray",
                "primitive": "int64",
                "inner_shape": [],
                "has_identifier": False,
                "parameters": {},
                "form_key": None,
            },
        },
        "has_identifier": False,
        "parameters": {},
        "form_key": None,
    }
    assert json.loads(ak.to_json(recordarray.as_tuple)) == [
        {"0": 1, "1": [1.1, 2.2, 3.3], "2": 1.1, "3": 1},
        {"0": 2, "1": [], "2": 2.2, "3": 2},
        {"0": 3, "1": [4.4, 5.5], "2": 3.3, "3": 3},
        {"0": 4, "1": [6.6], "2": 4.4, "3": 4},
        {"0": 5, "1": [7.7, 8.8, 9.9], "2": 5.5, "3": 5},
    ]


def test_scalar_record():
    content1 = ak.layout.NumpyArray(np.array([1, 2, 3, 4, 5], dtype=np.int64))
    content2 = ak.layout.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], dtype=np.float64)
    )
    offsets = ak.layout.Index64(np.array([0, 3, 3, 5, 6, 9], dtype=np.int64))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, content2)
    recordarray = ak.layout.RecordArray(
        [content1, listoffsetarray], keys=["one", "two"]
    )

    listoffsetarray = v1_to_v2(listoffsetarray)
    recordarray = v1_to_v2(recordarray)

    str(recordarray)
    str(recordarray[2])
    assert recordarray.typetracer[2].array.form == recordarray[2].array.form

    assert recordarray[2].fields == ["one", "two"]
    assert [ak.to_list(x) for x in recordarray[2].contents] == [3, [4.4, 5.5]]
    pairs = [(field, recordarray[2].content(field)) for field in recordarray[2].fields]
    assert pairs[0][0] == "one"
    assert pairs[1][0] == "two"
    assert pairs[0][1] == 3
    assert ak.to_list(pairs[1][1]) == [4.4, 5.5]
    assert ak.to_list(recordarray[2]) == {"one": 3, "two": [4.4, 5.5]}


def test_getitem():
    assert (
        ak._ext._slice_tostring((1, 2, [3], "four", ["five", "six"], slice(7, 8, 9)))
        == '[array([1]), array([2]), array([3]), "four", ["five", "six"], 7:8:9]'
    )

    content1 = ak.layout.NumpyArray(np.array([1, 2, 3, 4, 5], dtype=np.int64))
    content2 = ak.layout.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], dtype=np.float64)
    )
    offsets = ak.layout.Index64(np.array([0, 3, 3, 5, 6, 9], dtype=np.int64))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, content2)
    recordarray = ak.layout.RecordArray([content1, listoffsetarray, content2])
    recordarray = v1_to_v2(recordarray)

    assert ak.to_list(recordarray["2"]) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert recordarray.typetracer["2"].form == recordarray["2"].form
    assert ak.to_list(recordarray[["0", "1"]]) == [
        (1, [1.1, 2.2, 3.3]),
        (2, []),
        (3, [4.4, 5.5]),
        (4, [6.6]),
        (5, [7.7, 8.8, 9.9]),
    ]
    assert recordarray.typetracer[["0", "1"]].form == recordarray[["0", "1"]].form
    assert ak.to_list(recordarray[["1", "0"]]) == [
        ([1.1, 2.2, 3.3], 1),
        ([], 2),
        ([4.4, 5.5], 3),
        ([6.6], 4),
        ([7.7, 8.8, 9.9], 5),
    ]
    assert recordarray.typetracer[["1", "0"]].form == recordarray[["1", "0"]].form
    assert ak.to_list(recordarray[1:-1]) == [
        (2, [], 2.2),
        (3, [4.4, 5.5], 3.3),
        (4, [6.6], 4.4),
    ]
    assert recordarray.typetracer[1:-1].form == recordarray[1:-1].form
    assert ak.to_list(recordarray[2]) == (3, [4.4, 5.5], 3.3)
    assert recordarray.typetracer[2].array.form == recordarray[2].array.form
    assert ak.to_list(recordarray[2]["1"]) == [4.4, 5.5]
    assert recordarray.typetracer[2]["1"].form == recordarray[2]["1"].form
    assert ak.to_list(recordarray[2][["0", "1"]]) == (3, [4.4, 5.5])
    assert (
        recordarray.typetracer[2][["0", "1"]].array.form
        == recordarray[2][["0", "1"]].array.form
    )
    assert ak.to_list(recordarray[2][["1", "0"]]) == ([4.4, 5.5], 3)
    assert (
        recordarray.typetracer[2][["1", "0"]].array.form
        == recordarray[2][["1", "0"]].array.form
    )

    recordarray = ak.layout.RecordArray(
        {"one": content1, "two": listoffsetarray, "three": content2}
    )
    recordarray = v1_to_v2(recordarray)

    assert not recordarray.is_tuple

    assert ak.to_list(recordarray["three"]) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert recordarray.typetracer["three"].form == recordarray["three"].form
    assert ak.to_list(recordarray[["one", "two"]]) == [
        {"one": 1, "two": [1.1, 2.2, 3.3]},
        {"one": 2, "two": []},
        {"one": 3, "two": [4.4, 5.5]},
        {"one": 4, "two": [6.6]},
        {"one": 5, "two": [7.7, 8.8, 9.9]},
    ]
    assert (
        recordarray.typetracer[["one", "two"]].form == recordarray[["one", "two"]].form
    )
    assert ak.to_list(recordarray[["two", "one"]]) == [
        {"one": 1, "two": [1.1, 2.2, 3.3]},
        {"one": 2, "two": []},
        {"one": 3, "two": [4.4, 5.5]},
        {"one": 4, "two": [6.6]},
        {"one": 5, "two": [7.7, 8.8, 9.9]},
    ]
    assert (
        recordarray.typetracer[["two", "one"]].form == recordarray[["two", "one"]].form
    )
    assert ak.to_list(recordarray[1:-1]) == [
        {"one": 2, "two": [], "three": 2.2},
        {"one": 3, "two": [4.4, 5.5], "three": 3.3},
        {"one": 4, "two": [6.6], "three": 4.4},
    ]
    assert recordarray.typetracer[1:-1].form == recordarray[1:-1].form
    assert ak.to_list(recordarray[2]) == {"one": 3, "two": [4.4, 5.5], "three": 3.3}
    assert recordarray.typetracer[2].array.form == recordarray[2].array.form
    assert ak.to_list(recordarray[2]["two"]) == [4.4, 5.5]
    assert recordarray.typetracer[2]["two"].form == recordarray[2]["two"].form
    assert ak.to_list(recordarray[2][["one", "two"]]) == {"one": 3, "two": [4.4, 5.5]}
    assert (
        recordarray.typetracer[2][["one", "two"]].array.form
        == recordarray[2][["one", "two"]].array.form
    )
    assert ak.to_list(recordarray[2][["two", "one"]]) == {"one": 3, "two": [4.4, 5.5]}
    assert (
        recordarray.typetracer[2][["two", "one"]].array.form
        == recordarray[2][["two", "one"]].array.form
    )


def test_getitem_other_types():
    content1 = ak.layout.NumpyArray(np.array([1, 2, 3, 4, 5], dtype=np.int64))
    content2 = ak.layout.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], dtype=np.float64)
    )
    offsets1 = ak.layout.Index64(np.array([0, 3, 3, 5, 6, 9], dtype=np.int64))
    listoffsetarray1 = ak.layout.ListOffsetArray64(offsets1, content2)
    recordarray = ak.layout.RecordArray(
        {"one": content1, "two": listoffsetarray1, "three": content2}
    )

    offsets2 = ak.layout.Index64(np.array([0, 3, 3, 5], dtype=np.int64))
    listoffsetarray2 = ak.layout.ListOffsetArray64(offsets2, recordarray)

    listoffsetarray2 = v1_to_v2(listoffsetarray2)

    assert ak.to_list(listoffsetarray2["one"]) == [[1, 2, 3], [], [4, 5]]
    assert listoffsetarray2.typetracer["one"].form == listoffsetarray2["one"].form
    assert ak.to_list(listoffsetarray2["two"]) == [
        [[1.1, 2.2, 3.3], [], [4.4, 5.5]],
        [],
        [[6.6], [7.7, 8.8, 9.9]],
    ]
    assert listoffsetarray2.typetracer["two"].form == listoffsetarray2["two"].form
    assert ak.to_list(listoffsetarray2["three"]) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
    assert listoffsetarray2.typetracer["three"].form == listoffsetarray2["three"].form
    assert ak.to_list(listoffsetarray2[["two", "three"]]) == [
        [
            {"two": [1.1, 2.2, 3.3], "three": 1.1},
            {"two": [], "three": 2.2},
            {"two": [4.4, 5.5], "three": 3.3},
        ],
        [],
        [{"two": [6.6], "three": 4.4}, {"two": [7.7, 8.8, 9.9], "three": 5.5}],
    ]
    assert (
        listoffsetarray2.typetracer[["two", "three"]].form
        == listoffsetarray2[["two", "three"]].form
    )

    starts2 = ak.layout.Index64(np.array([0, 3, 3], dtype=np.int64))
    stops2 = ak.layout.Index64(np.array([3, 3, 5], dtype=np.int64))
    listarray2 = ak.layout.ListArray64(starts2, stops2, recordarray)
    listarray2 = v1_to_v2(listarray2)

    assert ak.to_list(listarray2["one"]) == [[1, 2, 3], [], [4, 5]]
    assert listarray2.typetracer["one"].form == listarray2["one"].form
    assert ak.to_list(listarray2["two"]) == [
        [[1.1, 2.2, 3.3], [], [4.4, 5.5]],
        [],
        [[6.6], [7.7, 8.8, 9.9]],
    ]
    assert listarray2.typetracer["two"].form == listarray2["two"].form
    assert ak.to_list(listarray2["three"]) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
    assert listarray2.typetracer["three"].form == listarray2["three"].form
    assert ak.to_list(listarray2[["two", "three"]]) == [
        [
            {"two": [1.1, 2.2, 3.3], "three": 1.1},
            {"two": [], "three": 2.2},
            {"two": [4.4, 5.5], "three": 3.3},
        ],
        [],
        [{"two": [6.6], "three": 4.4}, {"two": [7.7, 8.8, 9.9], "three": 5.5}],
    ]
    assert (
        listarray2.typetracer[["two", "three"]].form
        == listarray2[["two", "three"]].form
    )

    regulararray2 = ak.layout.RegularArray(recordarray, 1, zeros_length=0)
    regulararray2 = v1_to_v2(regulararray2)

    assert ak.to_list(regulararray2["one"]) == [[1], [2], [3], [4], [5]]
    assert regulararray2.typetracer["one"].form == regulararray2["one"].form
    assert ak.to_list(regulararray2["two"]) == [
        [[1.1, 2.2, 3.3]],
        [[]],
        [[4.4, 5.5]],
        [[6.6]],
        [[7.7, 8.8, 9.9]],
    ]
    assert regulararray2.typetracer["two"].form == regulararray2["two"].form
    assert ak.to_list(regulararray2["three"]) == [[1.1], [2.2], [3.3], [4.4], [5.5]]
    assert regulararray2.typetracer["three"].form == regulararray2["three"].form
    assert ak.to_list(regulararray2[["two", "three"]]) == [
        [{"two": [1.1, 2.2, 3.3], "three": 1.1}],
        [{"two": [], "three": 2.2}],
        [{"two": [4.4, 5.5], "three": 3.3}],
        [{"two": [6.6], "three": 4.4}],
        [{"two": [7.7, 8.8, 9.9], "three": 5.5}],
    ]
    assert (
        regulararray2.typetracer[["two", "three"]].form
        == regulararray2[["two", "three"]].form
    )


def test_getitem_next():
    content1 = ak.layout.NumpyArray(np.array([1, 2, 3, 4, 5], dtype=np.int64))
    content2 = ak.layout.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], dtype=np.float64)
    )
    content3 = ak.layout.NumpyArray(
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64)
    )
    offsets1 = ak.layout.Index64(np.array([0, 3, 3, 5, 6, 9], dtype=np.int64))
    listoffsetarray1 = ak.layout.ListOffsetArray64(offsets1, content2)
    listoffsetarray3 = ak.layout.ListOffsetArray64(offsets1, content3)
    recordarray = ak.layout.RecordArray(
        {
            "one": content1,
            "two": listoffsetarray1,
            "three": content2,
            "four": listoffsetarray3,
        }
    )
    offsets2 = ak.layout.Index64(np.array([0, 3, 3, 5], dtype=np.int64))
    listoffsetarray2 = ak.layout.ListOffsetArray64(offsets2, recordarray)

    listoffsetarray2 = v1_to_v2(listoffsetarray2)

    assert ak.to_list(listoffsetarray2[2, "one"]) == [4, 5]
    assert listoffsetarray2.typetracer[2, "one"].form == listoffsetarray2[2, "one"].form
    assert ak.to_list(listoffsetarray2[2, "two"]) == [[6.6], [7.7, 8.8, 9.9]]
    assert listoffsetarray2.typetracer[2, "two"].form == listoffsetarray2[2, "two"].form
    assert ak.to_list(listoffsetarray2[2, "three"]) == [4.4, 5.5]
    assert (
        listoffsetarray2.typetracer[2, "three"].form
        == listoffsetarray2[2, "three"].form
    )
    assert ak.to_list(listoffsetarray2[2, ["two", "three"]]) == [
        {"two": [6.6], "three": 4.4},
        {"two": [7.7, 8.8, 9.9], "three": 5.5},
    ]
    assert (
        listoffsetarray2.typetracer[2, ["two", "three"]].form
        == listoffsetarray2[2, ["two", "three"]].form
    )

    assert ak.to_list(listoffsetarray2[2, 1]) == {
        "one": 5,
        "two": [7.7, 8.8, 9.9],
        "three": 5.5,
        "four": [7, 8, 9],
    }
    assert (
        listoffsetarray2.typetracer[2, 1].array.form
        == listoffsetarray2[2, 1].array.form
    )
    with pytest.raises(IndexError):
        listoffsetarray2[2, 1, 0]
    assert listoffsetarray2[2, 1, "one"] == 5
    assert ak.to_list(listoffsetarray2[2, 1, "two"]) == [7.7, 8.8, 9.9]
    assert (
        listoffsetarray2.typetracer[2, 1, "two"].form
        == listoffsetarray2[2, 1, "two"].form
    )
    assert listoffsetarray2[2, 1, "two", 1] == 8.8
    assert ak.to_list(listoffsetarray2[2, 1, ["two", "four"], 1]) == {
        "two": 8.8,
        "four": 8,
    }
    assert (
        listoffsetarray2.typetracer[2, 1, ["two", "four"], 1].array.form
        == listoffsetarray2[2, 1, ["two", "four"], 1].array.form
    )
    assert ak.to_list(listoffsetarray2[2, 1, ["two", "four"], 1:]) == {
        "two": [8.8, 9.9],
        "four": [8, 9],
    }
