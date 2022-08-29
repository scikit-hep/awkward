# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    assert isinstance(ak._v2.from_iter((1, 2, 3)), ak._v2.Array)
    assert ak._v2.from_iter((1, 2, 3)).tolist() == [1, 2, 3]
    assert ak._v2.Array((1, 2, 3)).tolist() == [1, 2, 3]
    with pytest.raises(TypeError):
        ak._v2.Record((1, 2, 3))
    with pytest.raises(TypeError):
        ak._v2.Record([1, 2, 3])

    assert isinstance(ak._v2.from_iter({"one": 1, "two": 2}), ak._v2.Record)
    assert ak._v2.from_iter({"one": 1, "two": 2}).tolist() == {"one": 1, "two": 2}
    with pytest.raises(TypeError):
        ak._v2.Array({"one": 1, "two": 2})
    assert ak._v2.Record({"one": 1, "two": 2}).tolist() == {"one": 1, "two": 2}
