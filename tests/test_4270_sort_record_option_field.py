# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("stable", [True, False])
def test_sort_record_option_field_longer_than_record(dtype, ascending, stable):
    content = ak.contents.NumpyArray(np.array([3.0, 1.0, 2.0, 4.0], dtype=dtype))
    field = ak.contents.IndexedOptionArray(
        ak.index.Index64(np.array([0, -1, 1, 2, 3], dtype=np.int64)), content
    )
    array = ak.Array(ak.contents.RecordArray([field], ["x"], length=3))

    out = ak.sort(array, axis=0, ascending=ascending, stable=stable)

    values = [1.0, 3.0] if ascending else [3.0, 1.0]
    assert ak.to_list(out) == [{"x": values[0]}, {"x": values[1]}, {"x": None}]
    out_field = out.layout.content("x")
    assert isinstance(out_field, ak.contents.IndexedOptionArray)
    assert out_field.content.dtype == np.dtype(dtype)
    # Only two nonmissing values are sorted, but the option field still has
    # four nonmissing entries. Truncating the carry to the covered positions
    # must not truncate the content that its parent retains.
    assert out_field.content.length == content.length
