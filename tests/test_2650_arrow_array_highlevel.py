# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

pyarrow = pytest.importorskip("pyarrow")


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
