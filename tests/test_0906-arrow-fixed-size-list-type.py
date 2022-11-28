# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest

import awkward as ak

pyarrow = pytest.importorskip("pyarrow")

to_list = ak.operations.to_list


def test():
    import awkward._connect.pyarrow

    struct = pyarrow.struct(
        [
            pyarrow.field("x", pyarrow.list_(pyarrow.float64(), 2)),
            pyarrow.field("y", pyarrow.list_(pyarrow.float64(), 2)),
        ]
    )
    array = pyarrow.array(
        [
            {"x": [1.1, 2.1], "y": [3.1, 4.1]},
            {"x": [1.2, 2.2], "y": [3.2, 4.2]},
            {"x": [1.3, 2.3], "y": [3.3, 4.3]},
        ],
        type=struct,
    )
    as_awkward = awkward.from_arrow(array, highlevel=False)

    assert to_list(as_awkward) == [
        {"x": [1.1, 2.1], "y": [3.1, 4.1]},
        {"x": [1.2, 2.2], "y": [3.2, 4.2]},
        {"x": [1.3, 2.3], "y": [3.3, 4.3]},
    ]
    assert (
        str(as_awkward.form.type)
        == "{x: option[2 * ?float64], y: option[2 * ?float64]}"
    )
