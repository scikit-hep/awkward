# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import itertools

import numpy as np
import pytest

import awkward as ak


def test_reordered_index_list_of_records():
    """Issue #4316, first reproducer: a union whose list-of-records child is reordered."""
    z = ak.Array([[{"x": 1}], 2, []])
    p = z[[1, 2, 0]]
    q = ak.Array(p.tolist())

    assert p.tolist() == q.tolist()
    assert ak.array_equal(p, q)
    assert ak.array_equal(q, p)
    assert ak.array_equal(p, q, same_content_types=False)
    assert ak.almost_equal(p, q)


def _zero_field_union(index, offsets):
    return ak.Array(
        ak.contents.UnionArray(
            ak.index.Index8(np.array([1, 1], dtype=np.int8)),
            ak.index.Index64(np.array(index, dtype=np.int64)),
            [
                ak.contents.RecordArray([], [], 0),
                ak.contents.ListOffsetArray(
                    ak.index.Index64(np.array(offsets, dtype=np.int64)),
                    ak.contents.RecordArray([], [], 1),
                ),
            ],
        )
    )


def test_reordered_index_hand_built_union():
    """Issue #4316, second reproducer: identical forms, one union index reordered.

    `right` reaches the same values through a permuted union index, so
    `UnionArray.project` has to reorder the list child's content. The lazy
    carry that does so turns the child's `RecordArray` into an `IndexedArray`
    on one side only.
    """
    left = _zero_field_union([0, 1], [0, 1, 1])
    right = _zero_field_union([1, 0], [0, 0, 1])

    assert left.layout.form == right.layout.form
    assert left.tolist() == [[{}], []]
    assert right.tolist() == [[{}], []]
    assert ak.array_equal(left, right)
    assert ak.array_equal(right, left)
    assert ak.array_equal(left, right, same_content_types=False)
    assert ak.almost_equal(left, right)


@pytest.mark.parametrize(
    "data",
    [
        pytest.param([[1, 2], 3, []], id="list-of-numbers"),
        pytest.param([[{"x": 1}], 2, []], id="list-of-records"),
        pytest.param([[(1, 2.5)], 2, []], id="list-of-tuples"),
        pytest.param([[[1, 2]], 2, []], id="list-of-lists"),
        pytest.param([["one"], 2, []], id="list-of-strings"),
        pytest.param([[b"one"], 2, []], id="list-of-bytestrings"),
        pytest.param([[{"x": 1}, None], 2, []], id="list-of-optional-records"),
        pytest.param([[[{"x": 1}], []], 2, []], id="list-of-lists-of-records"),
        pytest.param(
            [[{"y": [1, 2]}, {"y": []}], 2, [{"y": [3]}]],
            id="list-of-records-of-lists",
        ),
        pytest.param([[{"s": "one"}], 2, [{"s": ""}]], id="list-of-records-of-strings"),
    ],
)
@pytest.mark.parametrize("permutation", list(itertools.permutations(range(3))))
def test_reordered_index_every_permutation(data, permutation):
    """Reordering a union's index must not change the outcome for any child type."""
    p = ak.Array(data)[list(permutation)]
    q = ak.Array(p.tolist())

    assert p.tolist() == q.tolist()
    assert ak.array_equal(p, q)
    assert ak.array_equal(q, p)
    assert ak.array_equal(p, q, same_content_types=False)
    assert ak.almost_equal(p, q)


def test_reordered_index_unequal_arrays():
    """A reordered union index must not make unequal arrays compare equal."""
    p = ak.Array([[{"x": 1}], 2, []])[[1, 2, 0]]
    assert p.tolist() == [2, [], [{"x": 1}]]

    different_value = ak.Array([2, [], [{"x": 999}]])
    assert not ak.array_equal(p, different_value)
    assert not ak.array_equal(different_value, p)
    assert not ak.almost_equal(p, different_value)

    different_lists = ak.Array([2, [{"x": 1}], []])
    assert not ak.array_equal(p, different_lists)
    assert not ak.array_equal(different_lists, p)
    assert not ak.almost_equal(p, different_lists)


def _union_of(list_content):
    return ak.Array(
        ak.contents.UnionArray(
            ak.index.Index8(np.array([0, 1, 1], dtype=np.int8)),
            ak.index.Index64(np.array([0, 1, 0], dtype=np.int64)),
            [
                ak.contents.NumpyArray(np.array([2], dtype=np.int64)),
                list_content,
            ],
        )
    )


def test_union_child_list_classes_still_compared():
    """`same_content_types=True` must still separate regular from variable lists."""
    var = _union_of(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 1, 2], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1, 3], dtype=np.int64)),
        )
    )
    regular = _union_of(
        ak.contents.RegularArray(
            ak.contents.NumpyArray(np.array([1, 3], dtype=np.int64)), 1
        )
    )

    assert var.tolist() == regular.tolist() == [2, [3], [1]]
    assert not ak.array_equal(var, regular)
    assert not ak.array_equal(regular, var)
    assert ak.array_equal(var, regular, check_regular=False)


def test_categorical_content_class_still_compared():
    """Packing keeps a categorical `IndexedArray`, so the class check must too."""
    categorical = ak.contents.IndexedArray(
        ak.index.Index64(np.array([0, 1, 0], dtype=np.int64)),
        ak.contents.NumpyArray(np.array([2, 3], dtype=np.int64)),
        parameters={"__array__": "categorical"},
    )
    plain = ak.contents.NumpyArray(np.array([2, 3, 2], dtype=np.int64))

    assert not ak.array_equal(categorical, plain)
    assert not ak.array_equal(plain, categorical)
    assert ak.array_equal(categorical, plain, same_content_types=False)
    assert ak.array_equal(categorical, categorical)
