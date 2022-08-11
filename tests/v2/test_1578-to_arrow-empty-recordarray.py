# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

pytest.importorskip("pyarrow")


def test():
    layout = ak._v2.contents.RecordArray([], fields=[], length=3)
    struct = ak._v2.to_arrow(layout)
    assert len(struct) == len(layout) == 3
    assert struct.tolist() == [{}, {}, {}]
