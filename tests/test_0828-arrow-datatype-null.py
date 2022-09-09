# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

pyarrow = pytest.importorskip("pyarrow")

to_list = ak._v2.operations.to_list


def test():
    import awkward._v2._connect.pyarrow

    assert to_list(
        awkward._v2._connect.pyarrow.handle_arrow(
            pyarrow.Table.from_pydict({"x": [None, None, None]})
        )
    ) == [{"x": None}, {"x": None}, {"x": None}]
    assert to_list(
        awkward._v2._connect.pyarrow.handle_arrow(
            pyarrow.Table.from_pydict({"x": [[None, None], [], [None]]})
        )
    ) == [{"x": [None, None]}, {"x": []}, {"x": [None]}]
