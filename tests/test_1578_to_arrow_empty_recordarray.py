# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest

import awkward as ak

pytest.importorskip("pyarrow")


def test():
    layout = ak.contents.RecordArray([], fields=[], length=3)
    struct = ak.to_arrow(layout)
    assert len(struct) == len(layout) == 3
    assert struct.tolist() == [{}, {}, {}]
