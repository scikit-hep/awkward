# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

pytest.importorskip("pyarrow")


def test():
    a = ak.highlevel.Array(
        ak.contents.RecordArray(
            [
                ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3])),
                ak.contents.NumpyArray(np.array([1, 2, 3])),
            ],
            ["x", "y"],
            parameters={"__record__": "Hello"},
        )
    )
    assert ak.operations.from_arrow(
        ak.operations.to_arrow_table(a)
    ).type.content.parameters == {"__record__": "Hello"}
