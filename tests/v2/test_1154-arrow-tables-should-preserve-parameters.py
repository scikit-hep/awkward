# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

pytest.importorskip("pyarrow")


def test():
    a = ak._v2.highlevel.Array(
        ak._v2.contents.RecordArray(
            [
                ak._v2.contents.NumpyArray(np.array([1.1, 2.2, 3.3])),
                ak._v2.contents.NumpyArray(np.array([1, 2, 3])),
            ],
            ["x", "y"],
            parameters={"__record__": "Hello"},
        )
    )
    assert ak._v2.operations.from_arrow(
        ak._v2.operations.to_arrow_table(a)
    ).type.content.parameters == {"__record__": "Hello"}
