# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
#
from __future__ import annotations

import pytest

safetensors = pytest.importorskip("safetensors")


def test_roundtrip(tmp_path):
    import awkward as ak

    array = ak.Array([[1, 2, 3], [], [4, 5], [6], [7, 8, 9, 10]])

    path = str(tmp_path / "test.safetensors")
    ak.to_safetensors(array, path)

    loaded = ak.from_safetensors(path)
    virtual_loaded = ak.from_safetensors(path, virtual=True)

    assert array.layout.is_equal_to(loaded.layout, all_parameters=True)
    assert array.layout.is_equal_to(
        virtual_loaded.layout.materialize(), all_parameters=True
    )


def test_virtual_array_to_safetensors(tmp_path):
    import awkward as ak

    array = ak.Array([[1, 2, 3], [], [4, 5], [6], [7, 8, 9, 10]])

    path0 = str(tmp_path / "test_virtual0.safetensors")
    path1 = str(tmp_path / "test_virtual1.safetensors")

    ak.to_safetensors(array, path0)
    virtual_loaded = ak.from_safetensors(path0, virtual=True)

    ak.to_safetensors(virtual_loaded, path1)
    loaded = ak.from_safetensors(path1, virtual=False)

    assert virtual_loaded.layout.materialize().is_equal_to(
        loaded.layout, all_parameters=True
    )
