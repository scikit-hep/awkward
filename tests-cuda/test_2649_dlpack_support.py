# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest
from packaging.version import parse as parse_version

import awkward as ak

if parse_version(np.__version__) < parse_version("1.23.0"):
    pytest.skip(
        "NumPy 1.23 or greater is required for DLPack testing", allow_module_level=True
    )


def test_from_dlpack_cupy():
    # This test only checks cupy usage, it doesn't explicitly test GPU & CPU
    cp = pytest.importorskip("cupy")
    cp_array = cp.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    array = ak.from_dlpack(cp_array)
    cp_from_ak = ak.to_cupy(array)
    assert cp.shares_memory(cp_array, cp_from_ak)
