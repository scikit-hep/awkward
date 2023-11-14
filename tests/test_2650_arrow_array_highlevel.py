# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest
from packaging.version import parse as parse_version

import awkward as ak

pyarrow = pytest.importorskip("pyarrow")
pyarrow_types = pytest.importorskip("pyarrow.types")


def test_simple_conversion():
    array = ak.Array(
        ak.contents.RecordArray(
            [
                ak.to_layout(np.arange(6, dtype=np.uint8)),
                ak.to_layout(np.arange(6, dtype=np.float64)),
            ],
            ["x", "y"],
        )
    )
    arrow_array = pyarrow.array(array)

    assert arrow_array.tolist() == array.to_list()


@pytest.mark.skipif(
    parse_version(pyarrow.__version__) < parse_version("12.0.0"),
    reason="pyarrow >= 12.0.0 required for casting test",
)
def test_type_cast():
    array = ak.mask(np.array([1, 2, 3], dtype=np.uint8), [True, False, False])
    arrow_array = pyarrow.array(array, type=pyarrow.float64())
    assert pyarrow_types.is_float64(arrow_array.type)

    arrow_array = pyarrow.array(array)
    assert pyarrow_types.is_uint8(arrow_array.type)


def test_zero_copy():
    np_array = np.array([1, 2, 3], dtype=np.uint32)
    ak_array = ak.from_numpy(np_array)
    arrow_array = pyarrow.array(ak_array)
    assert pyarrow_types.is_uint32(arrow_array.type)
    assert np.shares_memory(np_array, arrow_array.to_numpy())
