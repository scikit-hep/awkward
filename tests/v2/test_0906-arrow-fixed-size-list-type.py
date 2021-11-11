# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

pyarrow = pytest.importorskip("pyarrow")

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test():
    import awkward._v2._connect.pyarrow

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
    as_awkward = awkward._v2._connect.pyarrow.handle_arrow(array)

    assert ak.to_list(as_awkward) == [
        {"x": [1.1, 2.1], "y": [3.1, 4.1]},
        {"x": [1.2, 2.2], "y": [3.2, 4.2]},
        {"x": [1.3, 2.3], "y": [3.3, 4.3]},
    ]
    assert (
        str(as_awkward.form.type)
        == "{x: option[2 * ?float64], y: option[2 * ?float64]}"
    )
