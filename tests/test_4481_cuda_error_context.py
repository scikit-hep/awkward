# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import gc
import weakref

import numpy as np
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
        assert "layout=ListOffsetArray length=2" in snapshot.note
        assert "[1, 2]" not in snapshot.note
    del context, array, index
    gc.collect()
    assert all(ref() is None for ref in references)
    error = snapshot.decorate_exception(ValueError, ValueError("bad index"))
    assert "<Array layout=ListOffsetArray length=2>" in error.__notes__[0]
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


def test_cuda_capture_lowlevel_arguments_and_unknown_context():
    from awkward._errors import _cuda_argument

    layout = ak.to_layout([{"x": 1}])
    assert _cuda_argument(layout) == "<RecordArray length=1>"
    assert _cuda_argument(layout[0]) == "<Record>"
    assert "length=2 dtype=int64" in _cuda_argument(ak.index.Index64(np.array([0, 1])))
    assert _cuda_argument((1,)) == "(1,)"
    assert _cuda_argument(object()) == "<object>"
    assert len(_cuda_argument("x" * 1000)) == 74

    class CustomContext(ErrorContext):
        @property
        def note(self):
            raise AssertionError("custom formatting must not run during capture")

    with CustomContext():
        snapshot = ErrorContext.cuda_primary()
    assert "CustomContext(" in snapshot.note


@pytest.mark.thread_unsafe
@pytest.mark.parametrize("kind", ["array", "record"])
@pytest.mark.parametrize("fields", [1, 1000])
def test_cuda_capture_never_constructs_recursive_type(monkeypatch, kind, fields):
    array = ak.Array([{f"field{i}": [1, 2] for i in range(fields)}])
    value = array if kind == "array" else array[0]

    def forbidden(*args, **kwargs):
        raise AssertionError("CUDA capture must use outer metadata only")

    for cls in (ak.Array, ak.Record):
        monkeypatch.setattr(cls, "type", property(forbidden))
        monkeypatch.setattr(cls, "__repr__", forbidden)
        monkeypatch.setattr(cls, "__str__", forbidden)
    monkeypatch.setattr(ak.contents.RecordArray, "form", property(forbidden))
    with OperationErrorContext("ak.example", (value,), {}):
        snapshot = ErrorContext.cuda_primary()
    expected = (
        "<Array layout=RecordArray length=1>"
        if kind == "array"
        else "<Record layout=RecordArray>"
    )
    assert snapshot.args == [expected]
    assert len(snapshot.note) < 200
    assert "field0" not in snapshot.note


def test_cuda_capture_nested_metadata_and_unknown_length():
    from awkward._errors import _cuda_argument

    array = ak.Array([[1, 2], []])
    assert _cuda_argument([array]) == "[<Array layout=ListOffsetArray length=2>]"
    assert _cuda_argument({"pt": array}) == "<dict>"
    tracer = ak.Array(array.layout.to_typetracer(forget_length=True))
    assert _cuda_argument(tracer) == "<Array layout=ListOffsetArray length=##>"
