# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np

import awkward as ak
from awkward._errors import ErrorContext, _cuda_argument


def test_cuda_metadata_handles_lowlevel_and_selector_types():
    array = ak.Array([{"x": 1}])
    assert "length=1" in _cuda_argument(array.layout)
    assert _cuda_argument(array.layout[0]) == "<Record>"
    assert "Index64" in _cuda_argument(ak.index.Index64(np.array([0])))
    assert _cuda_argument((1,)) == "(1,)"
    assert _cuda_argument(b"x" * 100) == repr(b"x" * 72)
    assert _cuda_argument("x" * 100) == repr("x" * 72)
    assert _cuda_argument(object()) == "<object>"
    assert _cuda_argument([[[[1]]]]) == "[[[<list>]]]"
    assert _cuda_argument(None) == "None"


def test_unknown_cuda_context_uses_name_without_custom_formatting():
    class Custom(ErrorContext):
        @property
        def note(self):
            raise AssertionError("Custom context must not be formatted")

    with Custom():
        snapshot = ErrorContext.cuda_primary()
        assert "Custom(" in snapshot.note
        assert ErrorContext.cuda_primary() is snapshot
