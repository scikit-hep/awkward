# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
#
from __future__ import annotations

import os

import pytest

safetensors = pytest.importorskip("safetensors")


def test_roundtrip():
    import awkward as ak

    array = ak.Array([[1, 2, 3], [], [4, 5], [6], [7, 8, 9, 10]])

    path = "./test.safetensors"
    ak.to_safetensors(array, path)

    loaded = ak.from_safetensors(path)
    virtual_loaded = ak.from_safetensors(path, virtual=True)

    os.remove(path)

    assert array.layout.is_equal_to(loaded.layout, all_parameters=True)
    assert array.layout.is_equal_to(
        virtual_loaded.layout.materialize(), all_parameters=True
    )
