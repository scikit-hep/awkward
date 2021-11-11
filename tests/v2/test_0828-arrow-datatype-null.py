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

    assert ak.to_list(
        awkward._v2._connect.pyarrow.handle_arrow(
            pyarrow.Table.from_pydict({"x": [None, None, None]})
        )
    ) == [{"x": None}, {"x": None}, {"x": None}]
    assert ak.to_list(
        awkward._v2._connect.pyarrow.handle_arrow(
            pyarrow.Table.from_pydict({"x": [[None, None], [], [None]]})
        )
    ) == [{"x": [None, None]}, {"x": []}, {"x": [None]}]
