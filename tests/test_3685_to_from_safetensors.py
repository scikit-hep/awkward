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


def test_virtual_array_to_safetensors():
    import awkward as ak

    array = ak.Array([[1, 2, 3], [], [4, 5], [6], [7, 8, 9, 10]])

    path = "./test_virtual{}.safetensors".format

    ak.to_safetensors(array, path(0))
    virtual_loaded = ak.from_safetensors(path(0), virtual=True)

    ak.to_safetensors(virtual_loaded, path(1))
    loaded = ak.from_safetensors(path(1), virtual=False)

    os.remove(path(0))
    os.remove(path(1))

    assert virtual_loaded.layout.materialize().is_equal_to(
        loaded.layout, all_parameters=True
    )
