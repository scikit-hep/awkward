# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401

import awkward as ak


def test_indexed_of_union():
    unionarray = ak.from_iter(
        [0.0, 1.1, "zero", 2.2, "one", "two", "three", 3.3, 4.4, 5.5, "four"],
        highlevel=False,
    )
    indexedarray = ak.contents.IndexedArray.simplified(
        ak.index.Index64(np.array([4, 3, 3, 8, 7, 6], np.int64)),
        unionarray,
    )
    assert indexedarray.to_list() == ["one", 2.2, 2.2, 4.4, 3.3, "three"]


def test_indexedoption_of_union():
    unionarray = ak.from_iter(
        [0.0, 1.1, "zero", 2.2, "one", "two", "three", 3.3, 4.4, 5.5, "four"],
        highlevel=False,
    )
    indexedoptionarray = ak.contents.IndexedOptionArray.simplified(
        ak.index.Index64(np.array([-1, 4, 3, -1, 3, 8, 7, 6, -1], np.int64)),
        unionarray,
    )
    assert indexedoptionarray.to_list() == [
        None,
        "one",
        2.2,
        None,
        2.2,
        4.4,
        3.3,
        "three",
        None,
    ]


def test_indexedoption_of_union_of_option_1():
    unionarray = ak.contents.UnionArray(
        ak.index.Index8(np.array([0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1], dtype=np.int8)),
        ak.index.Index64(np.array([0, 1, 0, 2, 1, 2, 3, 3, 4, 5, 4], dtype=np.int64)),
        [
            ak.from_iter([0.0, 1.1, 2.2, 3.3, None, 5.5], highlevel=False),
            ak.from_iter(["zero", "one", "two", "three", "four"], highlevel=False),
        ],
    )
    indexedoptionarray = ak.contents.IndexedOptionArray.simplified(
        ak.index.Index64(np.array([-1, 4, 3, -1, 3, 8, 7, 6, -1], np.int64)),
        unionarray,
    )
    assert indexedoptionarray.to_list() == [
        None,
        "one",
        2.2,
        None,
        2.2,
        None,
        3.3,
        "three",
        None,
    ]


def test_indexedoption_of_union_of_option_2():
    unionarray = ak.contents.UnionArray(
        ak.index.Index8(np.array([0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1], dtype=np.int8)),
        ak.index.Index64(np.array([0, 1, 0, 2, 1, 2, 3, 3, 4, 5, 4], dtype=np.int64)),
        [
            ak.from_iter([0.0, 1.1, 2.2, 3.3, 4.4, 5.5], highlevel=False),
            ak.from_iter(["zero", None, "two", "three", "four"], highlevel=False),
        ],
    )
    indexedoptionarray = ak.contents.IndexedOptionArray.simplified(
        ak.index.Index64(np.array([-1, 4, 3, -1, 3, 8, 7, 6, -1], np.int64)),
        unionarray,
    )
    assert indexedoptionarray.to_list() == [
        None,
        None,
        2.2,
        None,
        2.2,
        4.4,
        3.3,
        "three",
        None,
    ]


def test_indexedoption_of_union_of_option_1_2():
    unionarray = ak.contents.UnionArray(
        ak.index.Index8(np.array([0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1], dtype=np.int8)),
        ak.index.Index64(np.array([0, 1, 0, 2, 1, 2, 3, 3, 4, 5, 4], dtype=np.int64)),
        [
            ak.from_iter([0.0, 1.1, 2.2, 3.3, None, 5.5], highlevel=False),
            ak.from_iter(["zero", None, "two", "three", "four"], highlevel=False),
        ],
    )
    indexedoptionarray = ak.contents.IndexedOptionArray.simplified(
        ak.index.Index64(np.array([-1, 4, 3, -1, 3, 8, 7, 6, -1], np.int64)),
        unionarray,
    )
    assert indexedoptionarray.to_list() == [
        None,
        None,
        2.2,
        None,
        2.2,
        None,
        3.3,
        "three",
        None,
    ]
