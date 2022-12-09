# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import json

import numpy as np
import pytest

import awkward as ak

to_list = ak.operations.to_list


def test_basic():
    content1 = ak.contents.NumpyArray(np.array([1, 2, 3, 4, 5], dtype=np.int64))
    content2 = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], dtype=np.float64)
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 9], dtype=np.int64))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content2)
    recordarray = ak.contents.RecordArray(
        [content1, listoffsetarray, content2, content1],
        fields=["one", "two", "2", "wonky"],
    )

    assert to_list(recordarray.content(0)) == [1, 2, 3, 4, 5]
    assert recordarray.to_typetracer().content(0).form == recordarray.content(0).form
    assert to_list(recordarray.content("two")) == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
        [6.6],
        [7.7, 8.8, 9.9],
    ]
    assert (
        recordarray.to_typetracer().content("two").form
        == recordarray.content("two").form
    )
    assert to_list(recordarray.content("wonky")) == [1, 2, 3, 4, 5]
    assert (
        recordarray.to_typetracer().content("wonky").form
        == recordarray.content("wonky").form
    )

    str(recordarray)

    assert json.loads(ak.forms.form.Form.to_json(recordarray.form)) == (
        {
            "class": "RecordArray",
            "fields": ["one", "two", "2", "wonky"],
            "contents": [
                {
                    "class": "NumpyArray",
                    "primitive": "int64",
                    "inner_shape": [],
                    "parameters": {},
                    "form_key": None,
                },
                {
                    "class": "ListOffsetArray",
                    "offsets": "i64",
                    "content": {
                        "class": "NumpyArray",
                        "primitive": "float64",
                        "inner_shape": [],
                        "parameters": {},
                        "form_key": None,
                    },
                    "parameters": {},
                    "form_key": None,
                },
                {
                    "class": "NumpyArray",
                    "primitive": "float64",
                    "inner_shape": [],
                    "parameters": {},
                    "form_key": None,
                },
                {
                    "class": "NumpyArray",
                    "primitive": "int64",
                    "inner_shape": [],
                    "parameters": {},
                    "form_key": None,
                },
            ],
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
    assert [to_list(x) for x in recordarray.contents] == [
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
    assert to_list(pairs[0][1]) == [1, 2, 3, 4, 5]
    assert to_list(pairs[1][1]) == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
        [6.6],
        [7.7, 8.8, 9.9],
    ]
    assert to_list(pairs[2][1]) == [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
    assert to_list(pairs[3][1]) == [1, 2, 3, 4, 5]

    assert json.loads(ak.forms.form.Form.to_json(recordarray.form)) == {
        "class": "RecordArray",
        "fields": ["one", "two", "2", "wonky"],
        "contents": [
            {
                "class": "NumpyArray",
                "primitive": "int64",
                "inner_shape": [],
                "parameters": {},
                "form_key": None,
            },
            {
                "class": "ListOffsetArray",
                "offsets": "i64",
                "content": {
                    "class": "NumpyArray",
                    "primitive": "float64",
                    "inner_shape": [],
                    "parameters": {},
                    "form_key": None,
                },
                "parameters": {},
                "form_key": None,
            },
            {
                "class": "NumpyArray",
                "primitive": "float64",
                "inner_shape": [],
                "parameters": {},
                "form_key": None,
            },
            {
                "class": "NumpyArray",
                "primitive": "int64",
                "inner_shape": [],
                "parameters": {},
                "form_key": None,
            },
        ],
        "parameters": {},
        "form_key": None,
    }


def test_basic_tofrom_json():
    content1 = ak.contents.NumpyArray(np.array([1, 2, 3, 4, 5], dtype=np.int64))
    content2 = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], dtype=np.float64)
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 9], dtype=np.int64))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content2)
    recordarray = ak.contents.RecordArray(
        [content1, listoffsetarray, content2, content1],
        fields=["one", "two", "2", "wonky"],
    )
    assert json.loads(ak.operations.to_json(recordarray.to_tuple())) == [
        {"0": 1, "1": [1.1, 2.2, 3.3], "2": 1.1, "3": 1},
        {"0": 2, "1": [], "2": 2.2, "3": 2},
        {"0": 3, "1": [4.4, 5.5], "2": 3.3, "3": 3},
        {"0": 4, "1": [6.6], "2": 4.4, "3": 4},
        {"0": 5, "1": [7.7, 8.8, 9.9], "2": 5.5, "3": 5},
    ]


def test_scalar_record():
    content1 = ak.contents.NumpyArray(np.array([1, 2, 3, 4, 5], dtype=np.int64))
    content2 = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], dtype=np.float64)
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 9], dtype=np.int64))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content2)
    recordarray = ak.contents.RecordArray(
        [content1, listoffsetarray], fields=["one", "two"]
    )

    str(recordarray)
    str(recordarray[2])
    assert recordarray.to_typetracer()[2].array.form == recordarray[2].array.form

    assert recordarray[2].fields == ["one", "two"]
    assert [to_list(x) for x in recordarray[2].contents] == [3, [4.4, 5.5]]
    pairs = [(field, recordarray[2].content(field)) for field in recordarray[2].fields]
    assert pairs[0][0] == "one"
    assert pairs[1][0] == "two"
    assert pairs[0][1] == 3
    assert to_list(pairs[1][1]) == [4.4, 5.5]
    assert to_list(recordarray[2]) == {"one": 3, "two": [4.4, 5.5]}


def test_type():
    content1 = ak.contents.NumpyArray(np.array([1, 2, 3, 4, 5], dtype=np.int64))
    content2 = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], dtype=np.float64)
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 9]))
    fields = None
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content2)
    recordarray = ak.contents.RecordArray([content1, listoffsetarray], fields)
    assert str(ak.operations.type(recordarray)) == "(int64, var * float64)"

    assert ak.operations.type(recordarray) == ak.types.RecordType(
        (
            ak.types.NumpyType("int64"),
            ak.types.ListType(ak.types.NumpyType("float64")),
        ),
        fields,
    )

    recordarray = ak.contents.RecordArray(
        [content1, listoffsetarray], fields=["one", "two"]
    )
    assert str(ak.operations.type(recordarray)) in (
        "{one: int64, two: var * float64}",
        "{two: var * float64, one: int64}",
    )

    assert (
        str(
            ak.types.RecordType(
                (ak.types.NumpyType("int32"), ak.types.NumpyType("float64")),
                None,
            )
        )
        == "(int32, float64)"
    )


def test_recordtype():
    content1 = ak.contents.NumpyArray(np.array([1, 2, 3, 4, 5], dtype=np.int64))
    content2 = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], dtype=np.float64)
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 9]))
    fields = None
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content2)
    recordarray = ak.contents.RecordArray([content1, listoffsetarray], fields)

    assert ak.operations.type(recordarray[2]) == ak.types.RecordType(
        (
            ak.types.NumpyType("int64"),
            ak.types.ListType(ak.types.NumpyType("float64")),
        ),
        None,
    )
    assert str(
        ak.types.RecordType(
            (
                ak.types.NumpyType("int32"),
                ak.types.NumpyType("float64"),
            ),
            ["one", "two"],
        )
    ) in ("{one: int32, two: float64}", "{two: float64, one: int32}")

    recordarray = ak.contents.RecordArray([content1, listoffsetarray], ["one", "two"])

    assert ak.operations.type(recordarray) == ak.types.RecordType(
        (
            ak.types.NumpyType("int64"),
            ak.types.ListType(ak.types.NumpyType("float64")),
        ),
        ["one", "two"],
    )
    assert ak.operations.type(recordarray[2]) == ak.types.RecordType(
        (
            ak.types.NumpyType("int64"),
            ak.types.ListType(ak.types.NumpyType("float64")),
        ),
        ["one", "two"],
    )


def test_getitem():
    content1 = ak.contents.NumpyArray(np.array([1, 2, 3, 4, 5], dtype=np.int64))
    content2 = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], dtype=np.float64)
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 9], dtype=np.int64))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content2)
    recordarray = ak.contents.RecordArray([content1, listoffsetarray, content2], None)

    assert to_list(recordarray["2"]) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert recordarray.to_typetracer()["2"].form == recordarray["2"].form
    assert to_list(recordarray[["0", "1"]]) == [
        (1, [1.1, 2.2, 3.3]),
        (2, []),
        (3, [4.4, 5.5]),
        (4, [6.6]),
        (5, [7.7, 8.8, 9.9]),
    ]
    assert recordarray.to_typetracer()[["0", "1"]].form == recordarray[["0", "1"]].form
    assert to_list(recordarray[["1", "0"]]) == [
        ([1.1, 2.2, 3.3], 1),
        ([], 2),
        ([4.4, 5.5], 3),
        ([6.6], 4),
        ([7.7, 8.8, 9.9], 5),
    ]
    assert recordarray.to_typetracer()[["1", "0"]].form == recordarray[["1", "0"]].form
    assert to_list(recordarray[1:-1]) == [
        (2, [], 2.2),
        (3, [4.4, 5.5], 3.3),
        (4, [6.6], 4.4),
    ]
    assert recordarray.to_typetracer()[1:-1].form == recordarray[1:-1].form
    assert to_list(recordarray[2]) == (3, [4.4, 5.5], 3.3)
    assert recordarray.to_typetracer()[2].array.form == recordarray[2].array.form
    assert to_list(recordarray[2]["1"]) == [4.4, 5.5]
    assert recordarray.to_typetracer()[2]["1"].form == recordarray[2]["1"].form
    assert to_list(recordarray[2][["0", "1"]]) == (3, [4.4, 5.5])
    assert (
        recordarray.to_typetracer()[2][["0", "1"]].array.form
        == recordarray[2][["0", "1"]].array.form
    )
    assert to_list(recordarray[2][["1", "0"]]) == ([4.4, 5.5], 3)
    assert (
        recordarray.to_typetracer()[2][["1", "0"]].array.form
        == recordarray[2][["1", "0"]].array.form
    )

    recordarray = ak.contents.RecordArray(
        [content1, listoffsetarray, content2], ["one", "two", "three"]
    )

    assert not recordarray.is_tuple

    assert to_list(recordarray["three"]) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert recordarray.to_typetracer()["three"].form == recordarray["three"].form
    assert to_list(recordarray[["one", "two"]]) == [
        {"one": 1, "two": [1.1, 2.2, 3.3]},
        {"one": 2, "two": []},
        {"one": 3, "two": [4.4, 5.5]},
        {"one": 4, "two": [6.6]},
        {"one": 5, "two": [7.7, 8.8, 9.9]},
    ]
    assert (
        recordarray.to_typetracer()[["one", "two"]].form
        == recordarray[["one", "two"]].form
    )
    assert to_list(recordarray[["two", "one"]]) == [
        {"one": 1, "two": [1.1, 2.2, 3.3]},
        {"one": 2, "two": []},
        {"one": 3, "two": [4.4, 5.5]},
        {"one": 4, "two": [6.6]},
        {"one": 5, "two": [7.7, 8.8, 9.9]},
    ]
    assert (
        recordarray.to_typetracer()[["two", "one"]].form
        == recordarray[["two", "one"]].form
    )
    assert to_list(recordarray[1:-1]) == [
        {"one": 2, "two": [], "three": 2.2},
        {"one": 3, "two": [4.4, 5.5], "three": 3.3},
        {"one": 4, "two": [6.6], "three": 4.4},
    ]
    assert recordarray.to_typetracer()[1:-1].form == recordarray[1:-1].form
    assert to_list(recordarray[2]) == {"one": 3, "two": [4.4, 5.5], "three": 3.3}
    assert recordarray.to_typetracer()[2].array.form == recordarray[2].array.form
    assert to_list(recordarray[2]["two"]) == [4.4, 5.5]
    assert recordarray.to_typetracer()[2]["two"].form == recordarray[2]["two"].form
    assert to_list(recordarray[2][["one", "two"]]) == {"one": 3, "two": [4.4, 5.5]}
    assert (
        recordarray.to_typetracer()[2][["one", "two"]].array.form
        == recordarray[2][["one", "two"]].array.form
    )
    assert to_list(recordarray[2][["two", "one"]]) == {"one": 3, "two": [4.4, 5.5]}
    assert (
        recordarray.to_typetracer()[2][["two", "one"]].array.form
        == recordarray[2][["two", "one"]].array.form
    )


def test_getitem_other_types():
    content1 = ak.contents.NumpyArray(np.array([1, 2, 3, 4, 5], dtype=np.int64))
    content2 = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], dtype=np.float64)
    )
    offsets1 = ak.index.Index64(np.array([0, 3, 3, 5, 6, 9], dtype=np.int64))
    listoffsetarray1 = ak.contents.ListOffsetArray(offsets1, content2)
    recordarray = ak.contents.RecordArray(
        [content1, listoffsetarray1, content2], ["one", "two", "three"]
    )

    offsets2 = ak.index.Index64(np.array([0, 3, 3, 5], dtype=np.int64))
    listoffsetarray2 = ak.contents.ListOffsetArray(offsets2, recordarray)

    assert to_list(listoffsetarray2["one"]) == [[1, 2, 3], [], [4, 5]]
    assert listoffsetarray2.to_typetracer()["one"].form == listoffsetarray2["one"].form
    assert to_list(listoffsetarray2["two"]) == [
        [[1.1, 2.2, 3.3], [], [4.4, 5.5]],
        [],
        [[6.6], [7.7, 8.8, 9.9]],
    ]
    assert listoffsetarray2.to_typetracer()["two"].form == listoffsetarray2["two"].form
    assert to_list(listoffsetarray2["three"]) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
    assert (
        listoffsetarray2.to_typetracer()["three"].form == listoffsetarray2["three"].form
    )
    assert to_list(listoffsetarray2[["two", "three"]]) == [
        [
            {"two": [1.1, 2.2, 3.3], "three": 1.1},
            {"two": [], "three": 2.2},
            {"two": [4.4, 5.5], "three": 3.3},
        ],
        [],
        [{"two": [6.6], "three": 4.4}, {"two": [7.7, 8.8, 9.9], "three": 5.5}],
    ]
    assert (
        listoffsetarray2.to_typetracer()[["two", "three"]].form
        == listoffsetarray2[["two", "three"]].form
    )

    starts2 = ak.index.Index64(np.array([0, 3, 3], dtype=np.int64))
    stops2 = ak.index.Index64(np.array([3, 3, 5], dtype=np.int64))
    listarray2 = ak.contents.ListArray(starts2, stops2, recordarray)

    assert to_list(listarray2["one"]) == [[1, 2, 3], [], [4, 5]]
    assert listarray2.to_typetracer()["one"].form == listarray2["one"].form
    assert to_list(listarray2["two"]) == [
        [[1.1, 2.2, 3.3], [], [4.4, 5.5]],
        [],
        [[6.6], [7.7, 8.8, 9.9]],
    ]
    assert listarray2.to_typetracer()["two"].form == listarray2["two"].form
    assert to_list(listarray2["three"]) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
    assert listarray2.to_typetracer()["three"].form == listarray2["three"].form
    assert to_list(listarray2[["two", "three"]]) == [
        [
            {"two": [1.1, 2.2, 3.3], "three": 1.1},
            {"two": [], "three": 2.2},
            {"two": [4.4, 5.5], "three": 3.3},
        ],
        [],
        [{"two": [6.6], "three": 4.4}, {"two": [7.7, 8.8, 9.9], "three": 5.5}],
    ]
    assert (
        listarray2.to_typetracer()[["two", "three"]].form
        == listarray2[["two", "three"]].form
    )

    regulararray2 = ak.contents.RegularArray(recordarray, 1, zeros_length=0)

    assert to_list(regulararray2["one"]) == [[1], [2], [3], [4], [5]]
    assert regulararray2.to_typetracer()["one"].form == regulararray2["one"].form
    assert to_list(regulararray2["two"]) == [
        [[1.1, 2.2, 3.3]],
        [[]],
        [[4.4, 5.5]],
        [[6.6]],
        [[7.7, 8.8, 9.9]],
    ]
    assert regulararray2.to_typetracer()["two"].form == regulararray2["two"].form
    assert to_list(regulararray2["three"]) == [[1.1], [2.2], [3.3], [4.4], [5.5]]
    assert regulararray2.to_typetracer()["three"].form == regulararray2["three"].form
    assert to_list(regulararray2[["two", "three"]]) == [
        [{"two": [1.1, 2.2, 3.3], "three": 1.1}],
        [{"two": [], "three": 2.2}],
        [{"two": [4.4, 5.5], "three": 3.3}],
        [{"two": [6.6], "three": 4.4}],
        [{"two": [7.7, 8.8, 9.9], "three": 5.5}],
    ]
    assert (
        regulararray2.to_typetracer()[["two", "three"]].form
        == regulararray2[["two", "three"]].form
    )


def test_getitem_next():
    content1 = ak.contents.NumpyArray(np.array([1, 2, 3, 4, 5], dtype=np.int64))
    content2 = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], dtype=np.float64)
    )
    content3 = ak.contents.NumpyArray(
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64)
    )
    offsets1 = ak.index.Index64(np.array([0, 3, 3, 5, 6, 9], dtype=np.int64))
    listoffsetarray1 = ak.contents.ListOffsetArray(offsets1, content2)
    listoffsetarray3 = ak.contents.ListOffsetArray(offsets1, content3)
    recordarray = ak.contents.RecordArray(
        [content1, listoffsetarray1, content2, listoffsetarray3],
        ["one", "two", "three", "four"],
    )
    offsets2 = ak.index.Index64(np.array([0, 3, 3, 5], dtype=np.int64))
    listoffsetarray2 = ak.contents.ListOffsetArray(offsets2, recordarray)

    assert to_list(listoffsetarray2[2, "one"]) == [4, 5]
    assert (
        listoffsetarray2.to_typetracer()[2, "one"].form
        == listoffsetarray2[2, "one"].form
    )
    assert to_list(listoffsetarray2[2, "two"]) == [[6.6], [7.7, 8.8, 9.9]]
    assert (
        listoffsetarray2.to_typetracer()[2, "two"].form
        == listoffsetarray2[2, "two"].form
    )
    assert to_list(listoffsetarray2[2, "three"]) == [4.4, 5.5]
    assert (
        listoffsetarray2.to_typetracer()[2, "three"].form
        == listoffsetarray2[2, "three"].form
    )
    assert to_list(listoffsetarray2[2, ["two", "three"]]) == [
        {"two": [6.6], "three": 4.4},
        {"two": [7.7, 8.8, 9.9], "three": 5.5},
    ]
    assert (
        listoffsetarray2.to_typetracer()[2, ["two", "three"]].form
        == listoffsetarray2[2, ["two", "three"]].form
    )

    assert to_list(listoffsetarray2[2, 1]) == {
        "one": 5,
        "two": [7.7, 8.8, 9.9],
        "three": 5.5,
        "four": [7, 8, 9],
    }
    assert (
        listoffsetarray2.to_typetracer()[2, 1].array.form
        == listoffsetarray2[2, 1].array.form
    )
    with pytest.raises(IndexError):
        listoffsetarray2[2, 1, 0]
    assert listoffsetarray2[2, 1, "one"] == 5
    assert to_list(listoffsetarray2[2, 1, "two"]) == [7.7, 8.8, 9.9]
    assert (
        listoffsetarray2.to_typetracer()[2, 1, "two"].form
        == listoffsetarray2[2, 1, "two"].form
    )
    assert listoffsetarray2[2, 1, "two", 1] == 8.8
    assert to_list(listoffsetarray2[2, 1, ["two", "four"], 1]) == {
        "two": 8.8,
        "four": 8,
    }
    assert (
        listoffsetarray2.to_typetracer()[2, 1, ["two", "four"], 1].array.form
        == listoffsetarray2[2, 1, ["two", "four"], 1].array.form
    )
    assert to_list(listoffsetarray2[2, 1, ["two", "four"], 1:]) == {
        "two": [8.8, 9.9],
        "four": [8, 9],
    }


def test_builder_tuple():
    builder = ak.highlevel.ArrayBuilder()
    assert str(builder.type) == "0 * unknown"
    assert builder.snapshot().to_list() == []

    builder.begin_tuple(0)
    builder.end_tuple()

    builder.begin_tuple(0)
    builder.end_tuple()

    builder.begin_tuple(0)
    builder.end_tuple()

    assert str(builder.type) == "3 * ()"
    assert builder.snapshot().to_list() == [(), (), ()]

    builder = ak.highlevel.ArrayBuilder()

    builder.begin_tuple(3)
    builder.index(0)
    builder.boolean(True)
    builder.index(1)
    builder.begin_list()
    builder.integer(1)
    builder.end_list()
    builder.index(2)
    builder.real(1.1)
    builder.end_tuple()

    builder.begin_tuple(3)
    builder.index(1)
    builder.begin_list()
    builder.integer(2)
    builder.integer(2)
    builder.end_list()
    builder.index(2)
    builder.real(2.2)
    builder.index(0)
    builder.boolean(False)
    builder.end_tuple()

    builder.begin_tuple(3)
    builder.index(2)
    builder.real(3.3)
    builder.index(1)
    builder.begin_list()
    builder.integer(3)
    builder.integer(3)
    builder.integer(3)
    builder.end_list()
    builder.index(0)
    builder.boolean(True)
    builder.end_tuple()

    assert str(builder.type) == "3 * (bool, var * int64, float64)"
    assert builder.snapshot().to_list() == [
        (True, [1], 1.1),
        (False, [2, 2], 2.2),
        (True, [3, 3, 3], 3.3),
    ]


def test_builder_record():
    builder = ak.highlevel.ArrayBuilder()
    assert str(builder.type) == "0 * unknown"
    assert builder.snapshot().to_list() == []

    builder.begin_record()
    builder.end_record()

    builder.begin_record()
    builder.end_record()

    builder.begin_record()
    builder.end_record()

    assert str(builder.type) == "3 * {}"
    assert builder.snapshot().to_list() == [{}, {}, {}]

    builder = ak.highlevel.ArrayBuilder()

    builder.begin_record()
    builder.field("one")
    builder.integer(1)
    builder.field("two")
    builder.real(1.1)
    builder.end_record()

    builder.begin_record()
    builder.field("two")
    builder.real(2.2)
    builder.field("one")
    builder.integer(2)
    builder.end_record()

    builder.begin_record()
    builder.field("one")
    builder.integer(3)
    builder.field("two")
    builder.real(3.3)
    builder.end_record()

    assert str(builder.type) == "3 * {one: int64, two: float64}"
    assert builder.snapshot().to_list() == [
        {"one": 1, "two": 1.1},
        {"one": 2, "two": 2.2},
        {"one": 3, "two": 3.3},
    ]


def test_fromiter():
    dataset = [
        [(1, 1.1), (2, 2.2), (3, 3.3)],
        [(1, [1.1, 2.2, 3.3]), (2, []), (3, [4.4, 5.5])],
        [[(1, 1.1), (2, 2.2), (3, 3.3)], [], [(4, 4.4), (5, 5.5)]],
        [((1, 1), 1.1), ((2, 2), 2.2), ((3, 3), 3.3)],
        [({"x": 1, "y": 1}, 1.1), ({"x": 2, "y": 2}, 2.2), ({"x": 3, "y": 3}, 3.3)],
        [{"one": 1, "two": 1.1}, {"one": 2, "two": 2.2}, {"one": 3, "two": 3.3}],
        [
            {"one": 1, "two": [1.1, 2.2, 3.3]},
            {"one": 2, "two": []},
            {"one": 3, "two": [4.4, 5.5]},
        ],
        [
            [{"one": 1, "two": 1.1}, {"one": 2, "two": 2.2}, {"one": 3, "two": 3.3}],
            [],
            [{"one": 4, "two": 4.4}, {"one": 5, "two": 5.5}],
        ],
        [
            {"one": {"x": 1, "y": 1}, "two": 1.1},
            {"one": {"x": 2, "y": 2}, "two": 2.2},
            {"one": {"x": 3, "y": 3}, "two": 3.3},
        ],
        [
            {"one": (1, 1), "two": 1.1},
            {"one": (2, 2), "two": 2.2},
            {"one": (3, 3), "two": 3.3},
        ],
    ]
    for datum in dataset:
        assert ak.to_list(ak.from_iter(datum)) == datum


def test_json():
    dataset = [
        '[{"one":1,"two":1.1},{"one":2,"two":2.2},{"one":3,"two":3.3}]',
        '[{"one":1,"two":[1.1,2.2,3.3]},{"one":2,"two":[]},{"one":3,"two":[4.4,5.5]}]',
        '[[{"one":1,"two":1.1},{"one":2,"two":2.2},{"one":3,"two":3.3}],[],[{"one":4,"two":4.4},{"one":5,"two":5.5}]]',
        '[{"one":{"x":1,"y":1},"two":1.1},{"one":{"x":2,"y":2},"two":2.2},{"one":{"x":3,"y":3},"two":3.3}]',
    ]
    for datum in dataset:
        assert ak.to_json(ak.from_json(datum)) == datum
