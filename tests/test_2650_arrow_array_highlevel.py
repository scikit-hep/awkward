# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

pyarrow = pytest.importorskip("pyarrow")
pyarrow_types = pytest.importorskip("pyarrow.types")


def test():
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


def test_type():
    array = ak.mask(np.array([1, 2, 3], dtype=np.uint8), [True, False, False])
    arrow_array = pyarrow.array(array, type=pyarrow.float64())
    assert pyarrow_types.is_float64(arrow_array.type)

    arrow_array = pyarrow.array(array)
    assert pyarrow_types.is_uint8(arrow_array.type)
