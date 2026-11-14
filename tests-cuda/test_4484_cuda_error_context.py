# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak
from awkward._errors import ErrorContext

cp = pytest.importorskip("cupy")


@pytest.mark.parametrize("option", [False, True])
@pytest.mark.parametrize("length", [32, 4096])
def test_invalid_field_projection_uses_metadata_note(monkeypatch, option, length):
    inner_type = ak.contents.IndexedOptionArray if option else ak.contents.IndexedArray
    inner = inner_type(
        ak.index.Index64(np.array([1, 0], dtype=np.int64)),
        ak.contents.NumpyArray(np.array([10, 20])),
    )
    indices = np.zeros(length, dtype=np.int64)
    indices[length // 2] = 2
    array = ak.to_backend(
        ak.Array(
            ak.contents.IndexedArray(
                ak.index.Index64(indices), ak.contents.RecordArray([inner], ["x"])
            )
        ),
        "cuda",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid CUDA projection must not format device values")

    monkeypatch.setattr(ErrorContext, "format_argument", forbidden)
    with pytest.raises(ValueError, match="index out of range") as caught:
        array["x"]
    assert any("type='" in note for note in caught.value.__notes__)
