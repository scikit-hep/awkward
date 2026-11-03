# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import gc
import weakref

import pytest

import awkward as ak
from awkward._errors import ErrorContext, OperationErrorContext, SlicingErrorContext


@pytest.mark.parametrize("kind", ["operation", "slice"])
def test_cuda_capture_no_value_formatting_or_buffer_retention(monkeypatch, kind):
    array = ak.Array([[1, 2], []])
    index = ak.Array([[0], []])
    references = [weakref.ref(array), weakref.ref(index)]

    def forbidden(*args, **kwargs):
        raise AssertionError("CUDA diagnostics must not format array values")

    monkeypatch.setattr(ak.Array, "_repr", forbidden)
    monkeypatch.setattr(ak.Array, "__repr__", forbidden)
    monkeypatch.setattr(ErrorContext, "format_argument", forbidden)
    if kind == "operation":
        context = OperationErrorContext(
            "ak.example", (array,), {"indices": index, "axis": 1}
        )
    else:
        context = SlicingErrorContext(array, (slice(None), index))
    with context:
        snapshot = ErrorContext.cuda_primary()
        assert ErrorContext.cuda_primary() is snapshot
        assert "var * int64" in snapshot.note
        assert "[1, 2]" not in snapshot.note
    del context, array, index
    gc.collect()
    assert all(ref() is None for ref in references)
    error = snapshot.decorate_exception(ValueError, ValueError("bad index"))
    assert "type='2 * var * int64'" in error.__notes__[0]
    if kind == "operation":
        assert "ak.example(" in error.__notes__[0]
        assert "axis = 1" in error.__notes__[0]
    else:
        assert "(:, <Array" in error.__notes__[0]


def test_cuda_capture_without_primary():
    assert ErrorContext.cuda_primary() is None


def test_cuda_capture_bounds_nested_selectors():
    selector = [list(range(100))] * 100
    context = SlicingErrorContext(ak.Array([1]), selector)
    with context:
        snapshot = ErrorContext.cuda_primary()
    assert len(snapshot.where) < 500
    assert "..." in snapshot.where
