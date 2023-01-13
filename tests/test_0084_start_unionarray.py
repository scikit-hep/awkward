# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

to_list = ak.operations.to_list


def test_getitem():
    content0 = ak.operations.from_iter(
        [[1.1, 2.2, 3.3], [], [4.4, 5.5]], highlevel=False
    )
    content1 = ak.operations.from_iter(
        ["one", "two", "three", "four", "five"], highlevel=False
    )
    tags = ak.index.Index8(np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=np.int8))

    backend = ak._backends.NumpyBackend.instance()
    array32 = ak.contents.UnionArray.regular_index(
        tags, index_cls=ak.index.Index32, backend=backend
    )
    arrayU32 = ak.contents.UnionArray.regular_index(
        tags, index_cls=ak.index.IndexU32, backend=backend
    )
    array64 = ak.contents.UnionArray.regular_index(
        tags, index_cls=ak.index.Index64, backend=backend
    )

    assert np.asarray(array32).tolist() == [
        0,
        1,
        0,
        1,
        2,
        2,
        3,
        4,
    ]
    assert np.asarray(array32).dtype == np.dtype(np.int32)
    assert np.asarray(arrayU32).tolist() == [
        0,
        1,
        0,
        1,
        2,
        2,
        3,
        4,
    ]
    assert np.asarray(arrayU32).dtype == np.dtype(np.uint32)
    assert np.asarray(array64).tolist() == [
        0,
        1,
        0,
        1,
        2,
        2,
        3,
        4,
    ]
    assert np.asarray(array64).dtype == np.dtype(np.int64)

    index = ak.index.Index(np.array([0, 1, 0, 1, 2, 2, 4, 3], dtype=np.int32))
    array = ak.contents.UnionArray(tags, index, [content0, content1])

    assert np.asarray(array.tags).tolist() == [1, 1, 0, 0, 1, 0, 1, 1]
    assert np.asarray(array.tags).dtype == np.dtype(np.int8)
    assert np.asarray(array.index).tolist() == [0, 1, 0, 1, 2, 2, 4, 3]
    assert np.asarray(array.index).dtype == np.dtype(np.int32)
    assert type(array.contents) is list
    assert [to_list(x) for x in array.contents] == [
        [[1.1, 2.2, 3.3], [], [4.4, 5.5]],
        ["one", "two", "three", "four", "five"],
    ]
    assert len(array.contents) == 2
    assert to_list(array.content(0)) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
    assert array.to_typetracer().content(0).form == array.content(0).form
    assert to_list(array.content(1)) == ["one", "two", "three", "four", "five"]
    assert array.to_typetracer().content(1).form == array.content(1).form
    assert to_list(array.project(0)) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
    assert array.to_typetracer().project(0).form == array.project(0).form
    assert to_list(array.project(1)) == ["one", "two", "three", "five", "four"]
    assert array.to_typetracer().project(1).form == array.project(1).form
    repr(array)

    assert to_list(array[0]) == "one"
    assert to_list(array[1]) == "two"
    assert to_list(array[2]) == [1.1, 2.2, 3.3]
    assert to_list(array[3]) == []
    assert to_list(array[4]) == "three"
    assert to_list(array[5]) == [4.4, 5.5]
    assert to_list(array[6]) == "five"
    assert to_list(array[7]) == "four"

    assert to_list(array) == [
        "one",
        "two",
        [1.1, 2.2, 3.3],
        [],
        "three",
        [4.4, 5.5],
        "five",
        "four",
    ]
    assert to_list(array[1:-1]) == [
        "two",
        [1.1, 2.2, 3.3],
        [],
        "three",
        [4.4, 5.5],
        "five",
    ]
    assert array.to_typetracer()[1:-1].form == array[1:-1].form
    assert to_list(array[2:-2]) == [[1.1, 2.2, 3.3], [], "three", [4.4, 5.5]]
    assert array.to_typetracer()[2:-2].form == array[2:-2].form
    assert to_list(array[::2]) == ["one", [1.1, 2.2, 3.3], "three", "five"]
    assert array.to_typetracer()[::2].form == array[::2].form
    assert to_list(array[::2, 1:]) == ["ne", [2.2, 3.3], "hree", "ive"]
    assert array.to_typetracer()[::2, 1:].form == array[::2, 1:].form
    assert to_list(array[:, :-1]) == [
        "on",
        "tw",
        [1.1, 2.2],
        [],
        "thre",
        [4.4],
        "fiv",
        "fou",
    ]
    assert array.to_typetracer()[:, :-1].form == array[:, :-1].form

    content2 = ak.operations.from_iter(
        [{"x": 0, "y": []}, {"x": 1, "y": [1.1]}, {"x": 2, "y": [1.1, 2.2]}],
        highlevel=False,
    )
    content3 = ak.operations.from_iter(
        [
            {"x": 0.0, "y": "zero", "z": False},
            {"x": 1.1, "y": "one", "z": True},
            {"x": 2.2, "y": "two", "z": False},
            {"x": 3.3, "y": "three", "z": True},
            {"x": 4.4, "y": "four", "z": False},
        ],
        highlevel=False,
    )
    array2 = ak.contents.UnionArray(tags, index, [content2, content3])

    assert to_list(array2) == [
        {"x": 0.0, "y": "zero", "z": False},
        {"x": 1.1, "y": "one", "z": True},
        {"x": 0, "y": []},
        {"x": 1, "y": [1.1]},
        {"x": 2.2, "y": "two", "z": False},
        {"x": 2, "y": [1.1, 2.2]},
        {"x": 4.4, "y": "four", "z": False},
        {"x": 3.3, "y": "three", "z": True},
    ]
    assert to_list(array2["x"]) == [0.0, 1.1, 0, 1, 2.2, 2, 4.4, 3.3]
    assert array2.to_typetracer()["x"].form == array2["x"].form
    assert to_list(array2["y"]) == [
        "zero",
        "one",
        [],
        [1.1],
        "two",
        [1.1, 2.2],
        "four",
        "three",
    ]
    assert array2.to_typetracer()["y"].form == array2["y"].form
    assert to_list(array2[:, "y", 1:]) == [
        "ero",
        "ne",
        [],
        [],
        "wo",
        [2.2],
        "our",
        "hree",
    ]
    assert array2.to_typetracer()[:, "y", 1:].form == array2[:, "y", 1:].form
    assert to_list(array2["y", :, 1:]) == [
        "ero",
        "ne",
        [],
        [],
        "wo",
        [2.2],
        "our",
        "hree",
    ]
    assert array2.to_typetracer()["y", :, 1:].form == array2["y", :, 1:].form
    with pytest.raises(IndexError) as err:
        array2[:, 1:, "y"]
    assert "cannot slice" in str(err.value)
    with pytest.raises(IndexError) as err:
        array2["z"]
    assert "no field 'z'" in str(err.value)

    array3 = ak.contents.UnionArray(tags, index, [content3, content2])
    array4 = ak.contents.UnionArray(
        tags, index, [content0, content1, content2, content3]
    )

    assert set(content2.fields) == {"x", "y"}
    assert set(content3.fields) == {"x", "y", "z"}
    assert set(array2.fields) == {"x", "y"}
    assert set(array3.fields) == {"x", "y"}
    assert (
        set(array4.fields) == set()
    )  # v2 definition: fields (old keys()) is now the INTERSECTION
