# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2, v1_to_v2_index

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test_getitem():
    content0 = ak.from_iter([[1.1, 2.2, 3.3], [], [4.4, 5.5]], highlevel=False)
    content1 = ak.from_iter(["one", "two", "three", "four", "five"], highlevel=False)
    tags = ak.layout.Index8(np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=np.int8))

    array32 = ak.layout.UnionArray8_32.regular_index(tags)
    arrayU32 = ak.layout.UnionArray8_U32.regular_index(tags)
    array64 = ak.layout.UnionArray8_64.regular_index(tags)

    array32 = v1_to_v2_index(array32)
    arrayU32 = v1_to_v2_index(arrayU32)
    array64 = v1_to_v2_index(array64)

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

    index = ak.layout.Index32(np.array([0, 1, 0, 1, 2, 2, 4, 3], dtype=np.int32))
    array = ak.layout.UnionArray8_32(tags, index, [content0, content1])

    array = v1_to_v2(array)

    assert np.asarray(array.tags).tolist() == [1, 1, 0, 0, 1, 0, 1, 1]
    assert np.asarray(array.tags).dtype == np.dtype(np.int8)
    assert np.asarray(array.index).tolist() == [0, 1, 0, 1, 2, 2, 4, 3]
    assert np.asarray(array.index).dtype == np.dtype(np.int32)
    assert type(array.contents) is list
    assert [ak.to_list(x) for x in array.contents] == [
        [[1.1, 2.2, 3.3], [], [4.4, 5.5]],
        ["one", "two", "three", "four", "five"],
    ]
    assert len(array.contents) == 2
    assert ak.to_list(array.content(0)) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
    assert array.typetracer.content(0).form == array.content(0).form
    assert ak.to_list(array.content(1)) == ["one", "two", "three", "four", "five"]
    assert array.typetracer.content(1).form == array.content(1).form
    assert ak.to_list(array.project(0)) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
    assert array.typetracer.project(0).form == array.project(0).form
    assert ak.to_list(array.project(1)) == ["one", "two", "three", "five", "four"]
    assert array.typetracer.project(1).form == array.project(1).form
    repr(array)

    assert ak.to_list(array[0]) == "one"
    assert ak.to_list(array[1]) == "two"
    assert ak.to_list(array[2]) == [1.1, 2.2, 3.3]
    assert ak.to_list(array[3]) == []
    assert ak.to_list(array[4]) == "three"
    assert ak.to_list(array[5]) == [4.4, 5.5]
    assert ak.to_list(array[6]) == "five"
    assert ak.to_list(array[7]) == "four"

    assert ak.to_list(array) == [
        "one",
        "two",
        [1.1, 2.2, 3.3],
        [],
        "three",
        [4.4, 5.5],
        "five",
        "four",
    ]
    assert ak.to_list(array[1:-1]) == [
        "two",
        [1.1, 2.2, 3.3],
        [],
        "three",
        [4.4, 5.5],
        "five",
    ]
    assert array.typetracer[1:-1].form == array[1:-1].form
    assert ak.to_list(array[2:-2]) == [[1.1, 2.2, 3.3], [], "three", [4.4, 5.5]]
    assert array.typetracer[2:-2].form == array[2:-2].form
    assert ak.to_list(array[::2]) == ["one", [1.1, 2.2, 3.3], "three", "five"]
    assert array.typetracer[::2].form == array[::2].form
    assert ak.to_list(array[::2, 1:]) == ["ne", [2.2, 3.3], "hree", "ive"]
    assert array.typetracer[::2, 1:].form == array[::2, 1:].form
    assert ak.to_list(array[:, :-1]) == [
        "on",
        "tw",
        [1.1, 2.2],
        [],
        "thre",
        [4.4],
        "fiv",
        "fou",
    ]
    assert array.typetracer[:, :-1].form == array[:, :-1].form

    content2 = ak.from_iter(
        [{"x": 0, "y": []}, {"x": 1, "y": [1.1]}, {"x": 2, "y": [1.1, 2.2]}],
        highlevel=False,
    )
    content3 = ak.from_iter(
        [
            {"x": 0.0, "y": "zero", "z": False},
            {"x": 1.1, "y": "one", "z": True},
            {"x": 2.2, "y": "two", "z": False},
            {"x": 3.3, "y": "three", "z": True},
            {"x": 4.4, "y": "four", "z": False},
        ],
        highlevel=False,
    )
    array2 = ak.layout.UnionArray8_32(tags, index, [content2, content3])

    array2 = v1_to_v2(array2)

    assert ak.to_list(array2) == [
        {"x": 0.0, "y": "zero", "z": False},
        {"x": 1.1, "y": "one", "z": True},
        {"x": 0, "y": []},
        {"x": 1, "y": [1.1]},
        {"x": 2.2, "y": "two", "z": False},
        {"x": 2, "y": [1.1, 2.2]},
        {"x": 4.4, "y": "four", "z": False},
        {"x": 3.3, "y": "three", "z": True},
    ]
    assert ak.to_list(array2["x"]) == [0.0, 1.1, 0, 1, 2.2, 2, 4.4, 3.3]
    assert array2.typetracer["x"].form == array2["x"].form
    assert ak.to_list(array2["y"]) == [
        "zero",
        "one",
        [],
        [1.1],
        "two",
        [1.1, 2.2],
        "four",
        "three",
    ]
    assert array2.typetracer["y"].form == array2["y"].form
    assert ak.to_list(array2[:, "y", 1:]) == [
        "ero",
        "ne",
        [],
        [],
        "wo",
        [2.2],
        "our",
        "hree",
    ]
    assert array2.typetracer[:, "y", 1:].form == array2[:, "y", 1:].form
    assert ak.to_list(array2["y", :, 1:]) == [
        "ero",
        "ne",
        [],
        [],
        "wo",
        [2.2],
        "our",
        "hree",
    ]
    assert array2.typetracer["y", :, 1:].form == array2["y", :, 1:].form
    with pytest.raises(IndexError) as err:
        array2[:, 1:, "y"]
    assert str(err.value).startswith("cannot slice")
    with pytest.raises(IndexError) as err:
        array2["z"]
    assert str(err.value).startswith("no field 'z'")

    array3 = ak.layout.UnionArray8_32(tags, index, [content3, content2])
    array4 = ak.layout.UnionArray8_32(
        tags, index, [content0, content1, content2, content3]
    )

    content2 = v1_to_v2(content2)
    content3 = v1_to_v2(content3)
    array3 = v1_to_v2(array3)
    array4 = v1_to_v2(array4)
    assert set(content2.fields) == set(["x", "y"])
    assert set(content3.fields) == set(["x", "y", "z"])
    assert set(array2.fields) == set(["x", "y"])
    assert set(array3.fields) == set(["x", "y"])
    assert (
        set(array4.fields) == set()
    )  # v2 definition: fields (old keys()) is now the INTERSECTION
