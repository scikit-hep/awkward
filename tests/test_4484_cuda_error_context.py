# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import pytest

import awkward as ak
from awkward._errors import ErrorContext, OperationErrorContext, SlicingErrorContext


@pytest.mark.parametrize("kind", ["operation", "slice"])
def test_immediate_cuda_error_reuses_snapshot(monkeypatch, kind):
    array = ak.Array([[1, 2], []])
    context = (
        OperationErrorContext("ak.example", (array,), {"axis": 1})
        if kind == "operation"
        else SlicingErrorContext(array, "x")
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Immediate CUDA errors must not format device values")

    monkeypatch.setattr(ErrorContext, "format_argument", forbidden)
    monkeypatch.setattr(SlicingErrorContext, "format_slice", forbidden)
    error = ValueError("index out of range in compiled CUDA code")
    with pytest.raises(ValueError) as caught:
        with context:
            snapshot = ErrorContext.cuda_primary()
            raise error
    assert caught.value is error
    assert caught.value.__notes__ == [snapshot.note]
    assert "2 * var * int64" in caught.value.__notes__[0]
    assert ErrorContext.primary() is None


def test_cpu_error_keeps_value_formatting():
    with pytest.raises(ValueError) as caught:
        with SlicingErrorContext(ak.Array([123, 456]), "x"):
            raise ValueError("bad field")
    assert "[123, 456]" in caught.value.__notes__[0]


def test_cuda_snapshot_preserves_assertion_guidance():
    with pytest.raises(AssertionError) as caught:
        with OperationErrorContext("ak.example", (), {}):
            ErrorContext.cuda_primary()
            raise AssertionError("internal error")
    assert "github.com/scikit-hep/awkward/issues" in caught.value.__notes__[0]
