# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest

import awkward as ak

pyarrow = pytest.importorskip("pyarrow")

to_list = ak.operations.to_list


def test():
    import awkward._connect.pyarrow

    assert to_list(
        awkward._connect.pyarrow.handle_arrow(
            pyarrow.Table.from_pydict({"x": [None, None, None]})
        )
    ) == [{"x": None}, {"x": None}, {"x": None}]
    assert to_list(
        awkward._connect.pyarrow.handle_arrow(
            pyarrow.Table.from_pydict({"x": [[None, None], [], [None]]})
        )
    ) == [{"x": [None, None]}, {"x": []}, {"x": [None]}]
