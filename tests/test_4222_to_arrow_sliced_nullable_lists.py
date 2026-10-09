# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

pytest.importorskip("pyarrow")


@pytest.mark.parametrize(
    "values", [[[0], [1, 2, 3], [4]], ["a", "bcd", "e"], [b"a", b"bcd", b"e"]]
)
@pytest.mark.parametrize(
    "index_type", [ak.index.Index32, ak.index.IndexU32, ak.index.Index64]
)
@pytest.mark.parametrize("extensionarray", [False, True])
@pytest.mark.parametrize("downsize", [False, True])
def test_sliced_nullable_lists(values, index_type, extensionarray, downsize, tmp_path):
    original = ak.Array(values).layout
    layout = ak.contents.ListOffsetArray(
        index_type(original.offsets.data),
        original.content,
        parameters=original.parameters,
    )
    array = ak.mask(ak.Array(layout)[1:], [True, False])
    expected = [values[1], None]
    options = {
        "extensionarray": extensionarray,
        "list_to32": downsize,
        "string_to32": downsize,
        "bytestring_to32": downsize,
    }

    assert array.to_list() == expected
    arrow = ak.to_arrow(array, **options)
    assert arrow.to_pylist() == expected
    assert ak.from_arrow(arrow).to_list() == expected
    path = tmp_path / "sliced.parquet"
    ak.to_parquet(array, path, **options)
    assert ak.from_parquet(path).to_list() == expected


def test_outer_slice_shifts_nullable_bytestring_offsets(tmp_path):
    inner = ak.Array([b"\x00", b"\x01", b"\x02"]).layout
    masked = ak.contents.ByteMaskedArray(
        ak.index.Index8(np.array([1, 1, 0], dtype=np.int8)), inner, valid_when=True
    )
    array = ak.Array(
        ak.contents.ListOffsetArray(ak.index.Index64(np.array([1, 3])), masked)
    )
    expected = [[b"\x01", None]]
    assert ak.to_arrow(array).to_pylist() == expected
    path = tmp_path / "nested.parquet"
    ak.to_parquet(array, path)
    assert ak.from_parquet(path).to_list() == expected


@pytest.mark.parametrize("mask", [[True, True], [False, False], [False, True]])
def test_sliced_mask_controls(mask):
    array = ak.mask(ak.Array([[0], [1, 2, 3], [4]])[1:], mask)
    assert ak.to_arrow(array).to_pylist() == array.to_list()
