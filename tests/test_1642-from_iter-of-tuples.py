# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    assert isinstance(ak.from_iter((1, 2, 3)), ak.Array)
    assert ak.from_iter((1, 2, 3)).tolist() == [1, 2, 3]
    assert ak.Array((1, 2, 3)).tolist() == [1, 2, 3]
    with pytest.raises(TypeError):
        ak.Record((1, 2, 3))
    with pytest.raises(TypeError):
        ak.Record([1, 2, 3])

    assert isinstance(ak.from_iter({"one": 1, "two": 2}), ak.Record)
    assert ak.from_iter({"one": 1, "two": 2}).tolist() == {"one": 1, "two": 2}
    with pytest.raises(TypeError):
        ak.Array({"one": 1, "two": 2})
    assert ak.Record({"one": 1, "two": 2}).tolist() == {"one": 1, "two": 2}
