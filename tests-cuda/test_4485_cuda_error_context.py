# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak
from awkward._errors import ErrorContext

cp = pytest.importorskip("cupy")


@pytest.mark.parametrize("length", [32, 4096])
def test_immediate_regularize_error_uses_metadata_note(monkeypatch, length):
    source = ak.to_backend(
        ak.Array(
            ak.contents.RegularArray(
                ak.contents.NumpyArray(np.arange(length * 8, dtype=np.int64)), 8
            )
        ),
        "cuda",
    )
    columns = ak.to_backend(ak.Array([0, 8]), "cuda")

    def forbidden(*args, **kwargs):
        raise AssertionError("Immediate CUDA errors must not format array values")

    monkeypatch.setattr(ErrorContext, "format_argument", forbidden)
    with pytest.raises((IndexError, ValueError), match="index out of range") as caught:
        source[:, columns]
    assert any("type='" in note for note in caught.value.__notes__)
