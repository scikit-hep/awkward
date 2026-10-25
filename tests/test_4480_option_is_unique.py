# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak
from awkward._kernels import NumpyKernel


@pytest.fixture(autouse=True)
def guard_native_boundaries(monkeypatch):
    # Fail safely on the regression instead of allowing native memory corruption.
    original = NumpyKernel.__call__

    def checked(kernel, *args):
        if kernel.key[0] == "awkward_sort":
            _, data, _, offsets, count, *_ = args
        elif kernel.key[0] in ("awkward_unique_ranges", "awkward_unique_ranges_bool"):
            data, offsets, count, _ = args
        else:
            return original(kernel, *args)
        assert np.all(offsets[:count] >= 0)
        assert np.all(offsets[:count] <= data.size)
        return original(kernel, *args)

    monkeypatch.setattr(NumpyKernel, "__call__", checked)


def optional_leaf(rows, representation):
    array = ak.Array(rows)
    content = array.layout.content
    if not content.is_option:
        content = ak.contents.IndexedOptionArray(
            ak.index.Index64(np.arange(content.length, dtype=np.int64)), content
        )
    if content.content.is_unknown:
        content = ak.contents.IndexedOptionArray(
            content.index, ak.contents.NumpyArray(np.empty(0, dtype=np.int64))
        )
    if representation == "indexed32":
        content = ak.contents.IndexedOptionArray(
            ak.index.Index32(content.index.data.astype(np.int32)), content.content
        )
    elif representation.startswith("byte"):
        content = content.to_ByteMaskedArray(valid_when=representation == "byte-valid")
    elif representation.startswith("bit"):
        content = content.to_BitMaskedArray(
            valid_when=representation == "bit-valid", lsb_order=True
        )
    return ak.contents.ListOffsetArray(array.layout.offsets, content)


@pytest.mark.parametrize(
    "representation",
    [
        "indexed64",
        "indexed32",
        "byte-valid",
        "byte-invalid",
        "bit-valid",
        "bit-invalid",
    ],
)
@pytest.mark.parametrize(
    "rows",
    [
        [[3, None, 3], [], [None, None], [1, None, 1]],
        [[3, None, 2], [], [None, None], [1, None, 2]],
        [[None, 1], [None, 2], [None, 3]],
        [[], [None, None], []],
        [[1, 2], [], [3, 4]],
        [[1, 1], [], [2]],
    ],
)
@pytest.mark.parametrize("axis", [-1, None])
def test_optional_boundaries(rows, representation, axis):
    layout = optional_leaf(rows, representation)
    groups = rows if axis == -1 else [[x for row in rows for x in row]]
    groups = [[x for x in row if x is not None] for row in groups]
    expected = all(len(row) == len(set(row)) for row in groups)
    assert ak._do.is_unique(layout, axis=axis) is expected


@pytest.mark.parametrize(
    "data,expected",
    [
        ([[1, 1]], False),
        ([[1], [1]], True),
        ([[1, 2], [], [3]], True),
        ([[True, True], [False]], False),
        ([[True, False], [], [True]], True),
        ([[1.5, 1.5]], False),
        ([[1.5], [], [1.5]], True),
    ],
)
def test_used_value_count(data, expected):
    assert ak._do.is_unique(ak.Array(data).layout, axis=-1) is expected


@pytest.mark.parametrize(
    "values,expected",
    [
        ([99, 99, 1, 2, 3, 4, 5, 88, 88], True),
        ([99, 99, 1, 2, 3, 4, 4, 88, 88], False),
    ],
)
@pytest.mark.parametrize("optional", [True, False])
def test_unused_prefix_and_tail(values, expected, optional):
    content = ak.contents.NumpyArray(np.array(values, dtype=np.int64))
    if optional:
        content = ak.contents.IndexedOptionArray(
            ak.index.Index64(np.array([0, -1, 2, 3, 4, 5, 6, -1, 8], dtype=np.int64)),
            content,
        )
    layout = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([2, 5, 5, 7], dtype=np.int64)), content
    )
    assert ak._do.is_unique(layout, axis=-1) is expected


@pytest.mark.parametrize(
    "values,expected",
    [([], True), ([None, None], True), ([1, None, 1], False), ([1, None, 2], True)],
)
@pytest.mark.parametrize("axis", [None, -1])
def test_flat_optional(values, expected, axis):
    assert ak._do.is_unique(ak.Array(values).layout, axis=axis) is expected


@pytest.mark.parametrize(
    "values,expected",
    [
        ([None, [[1, None, 1]], [[], [None, None]]], False),
        ([None, [[1, None, 2]], [[], [None, None]]], True),
        ([[1, None, 1], None, [2, None]], False),
        ([[1, None, 2], None, [2, None]], True),
    ],
)
def test_nested_missing_lists(values, expected):
    assert ak._do.is_unique(ak.Array(values).layout, axis=-1) is expected
