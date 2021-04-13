# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


pyarrow = pytest.importorskip("pyarrow")


def test():
    assert ak.from_arrow(
        pyarrow.Table.from_pydict({"x": [None, None, None]})
    ).tolist() == [{"x": None}, {"x": None}, {"x": None}]
    assert ak.from_arrow(
        pyarrow.Table.from_pydict({"x": [[None, None], [], [None]]})
    ).tolist() == [{"x": [None, None]}, {"x": []}, {"x": [None]}]
