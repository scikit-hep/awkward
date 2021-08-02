# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2


def test_basic():
    content1 = ak.layout.NumpyArray(np.array([1, 2, 3, 4, 5]))
    content2 = ak.layout.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak.layout.Index64(np.array([0, 3, 3, 5, 6, 9]))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, content2)
    recordarray = ak.layout.RecordArray(
        [content1, listoffsetarray, content2, content1],
        keys=["one", "two", "2", "wonky"],
    )

    recordarray = v1_to_v2(recordarray)

    assert ak.to_list(recordarray.content(0)) == [1, 2, 3, 4, 5]
    assert ak.to_list(recordarray.content("two")) == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
        [6.6],
        [7.7, 8.8, 9.9],
    ]
    assert ak.to_list(recordarray.content("wonky")) == [1, 2, 3, 4, 5]

    str(recordarray)
    assert ak._v2.forms.form.Form.to_json(recordarray.form) == (
        '{"class": "RecordArray", "contents": {"one": {"class": "NumpyArray", '
        '"primitive": "int64", "inner_shape": [], "has_identifier": false, '
        '"parameters": {}, "form_key": null}, "two": {"class": "ListOffsetArray", '
        '"offsets": "i64", "content": {"class": "NumpyArray", "primitive": "float64", '
        '"inner_shape": [], "has_identifier": false, "parameters": {}, "form_key": '
        'null}, "has_identifier": false, "parameters": {}, "form_key": null}, "2": '
        '{"class": "NumpyArray", "primitive": "float64", "inner_shape": [], '
        '"has_identifier": false, "parameters": {}, "form_key": null}, "wonky": '
        '{"class": "NumpyArray", "primitive": "int64", "inner_shape": [], '
        '"has_identifier": false, "parameters": {}, "form_key": null}}, '
        '"has_identifier": false, "parameters": {}, "form_key": null}'
    )

    assert len(recordarray) == 5
    assert recordarray.index_to_key(0) == "one"
    assert recordarray.index_to_key(1) == "two"
    assert recordarray.index_to_key(2) == "2"
    assert recordarray.index_to_key(3) == "wonky"
    assert recordarray.key_to_index("wonky") == 3
    assert recordarray.key_to_index("one") == 0
    # FIXME?
    # assert recordarray.key_to_index("0") == 0
    assert recordarray.key_to_index("two") == 1
    # assert recordarray.key_to_index("1") == 1
    assert recordarray.key_to_index("2") == 2
    assert recordarray.haskey("wonky")
    assert recordarray.haskey("one")
    # assert recordarray.haskey("0")
    assert recordarray.haskey("two")
    # assert recordarray.haskey("1")
    assert recordarray.haskey("2")

    assert recordarray.keys == ["one", "two", "2", "wonky"]
    assert [ak.to_list(x) for x in recordarray.contents] == [
        [1, 2, 3, 4, 5],
        [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]],
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9],
        [1, 2, 3, 4, 5],
    ]
    pairs = recordarray.contentitems()
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

    # assert (
    #     ak._v2.forms.form.Form.to_json(recordarray.form)
    #     == '{"class": "RecordArray", "contents": {"one": {"class": "NumpyArray", '
    #     '"primitive": "int64", "inner_shape": [], "has_identifier": false, '
    #     '"parameters": {}, "form_key": null}, "two": {"class": "ListOffsetArray", '
    #     '"offsets": "i64", "content": {"class": "NumpyArray", "primitive": "float64", '
    #     '"inner_shape": [], "has_identifier": false, "parameters": {}, "form_key": '
    #     'null}, "has_identifier": false, "parameters": {}, "form_key": null}, "2": '
    #     '{"class": "NumpyArray", "primitive": "float64", "inner_shape": [], '
    #     '"has_identifier": false, "parameters": {}, "form_key": null}, "wonky": '
    #     '{"class": "NumpyArray", "primitive": "int64", "inner_shape": [], '
    #     '"has_identifier": false, "parameters": {}, "form_key": null}}, '
    #     '"has_identifier": false, "parameters": {}, "form_key": null}'
    # )
    # FIXME
    # assert (
    #     ak.to_json(recordarray.astuple)
    #     == '[{"0":1,"1":[1.1,2.2,3.3],"2":1.1,"3":1},{"0":2,"1":[],"2":2.2,"3":2},{"0":3,"1":[4.4,5.5],"2":3.3,"3":3},{"0":4,"1":[6.6],"2":4.4,"3":4},{"0":5,"1":[7.7,8.8,9.9],"2":5.5,"3":5}]'
    # )


def test_scalar_record():
    content1 = ak.layout.NumpyArray(np.array([1, 2, 3, 4, 5]))
    content2 = ak.layout.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak.layout.Index64(np.array([0, 3, 3, 5, 6, 9]))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, content2)
    recordarray = ak.layout.RecordArray(
        [content1, listoffsetarray], keys=["one", "two"]
    )
    listoffsetarray = v1_to_v2(listoffsetarray)
    recordarray = v1_to_v2(recordarray)

    str(recordarray)
    str(recordarray[2])

    # FIXME
    # assert ak._v2.forms.form.Form.to_json(recordarray[2].form) == '{"one":3,"two":[4.4,5.5]}'

    # assert recordarray[2].keys == ["one", "two"]
    # assert [ak.to_list(x) for x in recordarray[2]] == [3, [4.4, 5.5]]
    # pairs = recordarray[2].contentitems()
    # assert pairs[0][0] == "one"
    # assert pairs[1][0] == "two"
    # assert pairs[0][1] == 3
    # assert ak.to_list(pairs[1][1]) == [4.4, 5.5]
    # assert ak.to_list(recordarray[2]) == {"one": 3, "two": [4.4, 5.5]}

    # assert ak.to_list(ak.layout.Record(recordarray, 2)) == {"one": 3, "two": [4.4, 5.5]}


def test_getitem():
    assert (
        ak._ext._slice_tostring((1, 2, [3], "four", ["five", "six"], slice(7, 8, 9)))
        == '[array([1]), array([2]), array([3]), "four", ["five", "six"], 7:8:9]'
    )

    content1 = ak.layout.NumpyArray(np.array([1, 2, 3, 4, 5], dtype=np.int64))
    content2 = ak.layout.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], dtype=np.float64)
    )
    offsets = ak.layout.Index64(np.array([0, 3, 3, 5, 6, 9]))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, content2)
    recordarray = ak.layout.RecordArray([content1, listoffsetarray, content2])

    recordarray = v1_to_v2(recordarray)
    # FIXME
    # assert recordarray.istuple

    assert ak.to_list(recordarray["2"]) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert ak.to_list(recordarray[["0", "1"]]) == [
        (1, [1.1, 2.2, 3.3]),
        (2, []),
        (3, [4.4, 5.5]),
        (4, [6.6]),
        (5, [7.7, 8.8, 9.9]),
    ]
    assert ak.to_list(recordarray[["1", "0"]]) == [
        ([1.1, 2.2, 3.3], 1),
        ([], 2),
        ([4.4, 5.5], 3),
        ([6.6], 4),
        ([7.7, 8.8, 9.9], 5),
    ]
    assert ak.to_list(recordarray[1:-1]) == [
        (2, [], 2.2),
        (3, [4.4, 5.5], 3.3),
        (4, [6.6], 4.4),
    ]
    assert ak.to_list(recordarray[2]) == (3, [4.4, 5.5], 3.3)
    assert ak.to_list(recordarray[2]["1"]) == [4.4, 5.5]
    assert ak.to_list(recordarray[2][["0", "1"]]) == (3, [4.4, 5.5])
    assert ak.to_list(recordarray[2][["1", "0"]]) == ([4.4, 5.5], 3)

    recordarray = ak.layout.RecordArray(
        {"one": content1, "two": listoffsetarray, "three": content2}
    )
    assert not recordarray.istuple

    assert ak.to_list(recordarray["three"]) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert ak.to_list(recordarray[["one", "two"]]) == [
        {"one": 1, "two": [1.1, 2.2, 3.3]},
        {"one": 2, "two": []},
        {"one": 3, "two": [4.4, 5.5]},
        {"one": 4, "two": [6.6]},
        {"one": 5, "two": [7.7, 8.8, 9.9]},
    ]
    assert ak.to_list(recordarray[["two", "one"]]) == [
        {"one": 1, "two": [1.1, 2.2, 3.3]},
        {"one": 2, "two": []},
        {"one": 3, "two": [4.4, 5.5]},
        {"one": 4, "two": [6.6]},
        {"one": 5, "two": [7.7, 8.8, 9.9]},
    ]
    assert ak.to_list(recordarray[1:-1]) == [
        {"one": 2, "two": [], "three": 2.2},
        {"one": 3, "two": [4.4, 5.5], "three": 3.3},
        {"one": 4, "two": [6.6], "three": 4.4},
    ]
    assert ak.to_list(recordarray[2]) == {"one": 3, "two": [4.4, 5.5], "three": 3.3}
    assert ak.to_list(recordarray[2]["two"]) == [4.4, 5.5]
    assert ak.to_list(recordarray[2][["one", "two"]]) == {"one": 3, "two": [4.4, 5.5]}
    assert ak.to_list(recordarray[2][["two", "one"]]) == {"one": 3, "two": [4.4, 5.5]}


def test_getitem_other_types():
    content1 = ak.layout.NumpyArray(np.array([1, 2, 3, 4, 5], dtype=np.int64))
    content2 = ak.layout.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], dtype=np.float64)
    )
    offsets1 = ak.layout.Index64(np.array([0, 3, 3, 5, 6, 9]))
    listoffsetarray1 = ak.layout.ListOffsetArray64(offsets1, content2)
    recordarray = ak.layout.RecordArray(
        {"one": content1, "two": listoffsetarray1, "three": content2}
    )

    offsets2 = ak.layout.Index64(np.array([0, 3, 3, 5]))
    listoffsetarray2 = ak.layout.ListOffsetArray64(offsets2, recordarray)

    listoffsetarray2 = v1_to_v2(listoffsetarray2)

    assert ak.to_list(listoffsetarray2["one"]) == [[1, 2, 3], [], [4, 5]]
    assert ak.to_list(listoffsetarray2["two"]) == [
        [[1.1, 2.2, 3.3], [], [4.4, 5.5]],
        [],
        [[6.6], [7.7, 8.8, 9.9]],
    ]
    assert ak.to_list(listoffsetarray2["three"]) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
    assert ak.to_list(listoffsetarray2[["two", "three"]]) == [
        [
            {"two": [1.1, 2.2, 3.3], "three": 1.1},
            {"two": [], "three": 2.2},
            {"two": [4.4, 5.5], "three": 3.3},
        ],
        [],
        [{"two": [6.6], "three": 4.4}, {"two": [7.7, 8.8, 9.9], "three": 5.5}],
    ]

    starts2 = ak.layout.Index64(np.array([0, 3, 3]))
    stops2 = ak.layout.Index64(np.array([3, 3, 5]))
    listarray2 = ak.layout.ListArray64(starts2, stops2, recordarray)
    assert ak.to_list(listarray2["one"]) == [[1, 2, 3], [], [4, 5]]
    assert ak.to_list(listarray2["two"]) == [
        [[1.1, 2.2, 3.3], [], [4.4, 5.5]],
        [],
        [[6.6], [7.7, 8.8, 9.9]],
    ]
    assert ak.to_list(listarray2["three"]) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
    assert ak.to_list(listarray2[["two", "three"]]) == [
        [
            {"two": [1.1, 2.2, 3.3], "three": 1.1},
            {"two": [], "three": 2.2},
            {"two": [4.4, 5.5], "three": 3.3},
        ],
        [],
        [{"two": [6.6], "three": 4.4}, {"two": [7.7, 8.8, 9.9], "three": 5.5}],
    ]

    regulararray2 = ak.layout.RegularArray(recordarray, 1, zeros_length=0)
    assert ak.to_list(regulararray2["one"]) == [[1], [2], [3], [4], [5]]
    assert ak.to_list(regulararray2["two"]) == [
        [[1.1, 2.2, 3.3]],
        [[]],
        [[4.4, 5.5]],
        [[6.6]],
        [[7.7, 8.8, 9.9]],
    ]
    assert ak.to_list(regulararray2["three"]) == [[1.1], [2.2], [3.3], [4.4], [5.5]]
    assert ak.to_list(regulararray2[["two", "three"]]) == [
        [{"two": [1.1, 2.2, 3.3], "three": 1.1}],
        [{"two": [], "three": 2.2}],
        [{"two": [4.4, 5.5], "three": 3.3}],
        [{"two": [6.6], "three": 4.4}],
        [{"two": [7.7, 8.8, 9.9], "three": 5.5}],
    ]


def test_getitem_next():
    content1 = ak.layout.NumpyArray(np.array([1, 2, 3, 4, 5], dtype=np.int64))
    content2 = ak.layout.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], dtype=np.float64)
    )
    content3 = ak.layout.NumpyArray(
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64)
    )
    offsets1 = ak.layout.Index64(np.array([0, 3, 3, 5, 6, 9]))
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
    offsets2 = ak.layout.Index64(np.array([0, 3, 3, 5]))
    listoffsetarray2 = ak.layout.ListOffsetArray64(offsets2, recordarray)

    listoffsetarray2 = v1_to_v2(listoffsetarray2)

    assert ak.to_list(listoffsetarray2[2, "one"]) == [4, 5]
    assert ak.to_list(listoffsetarray2[2, "two"]) == [[6.6], [7.7, 8.8, 9.9]]
    assert ak.to_list(listoffsetarray2[2, "three"]) == [4.4, 5.5]
    assert ak.to_list(listoffsetarray2[2, ["two", "three"]]) == [
        {"two": [6.6], "three": 4.4},
        {"two": [7.7, 8.8, 9.9], "three": 5.5},
    ]

    assert ak.to_list(listoffsetarray2[2, 1]) == {
        "one": 5,
        "two": [7.7, 8.8, 9.9],
        "three": 5.5,
        "four": [7, 8, 9],
    }
    with pytest.raises(IndexError):
        listoffsetarray2[2, 1, 0]
    assert listoffsetarray2[2, 1, "one"] == 5
    assert ak.to_list(listoffsetarray2[2, 1, "two"]) == [7.7, 8.8, 9.9]
    assert listoffsetarray2[2, 1, "two", 1] == 8.8
    assert ak.to_list(listoffsetarray2[2, 1, ["two", "four"], 1]) == {
        "two": 8.8,
        "four": 8,
    }
    assert ak.to_list(listoffsetarray2[2, 1, ["two", "four"], 1:]) == {
        "two": [8.8, 9.9],
        "four": [8, 9],
    }
