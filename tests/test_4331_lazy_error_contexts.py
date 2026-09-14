# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import gc
import weakref

import pytest

import awkward as ak
from awkward._errors import OperationErrorContext, SlicingErrorContext
from awkward._nplikes.numpy import Numpy
from awkward.errors import AxisError


def test_operation_note_still_reports_name_args_and_kwargs():
    with pytest.raises(AxisError) as excinfo:
        ak.num(ak.Array([1, 2, 3]), axis=1)

    (note,) = excinfo.value.__notes__
    assert "This error occurred while calling" in note
    assert "ak.num(" in note
    assert "<Array [1, 2, 3] type='3 * int64'>" in note
    assert "axis = 1" in note


def test_slicing_note_still_reports_array_and_slice():
    with pytest.raises(IndexError) as excinfo:
        ak.Array([[1, 2, 3], [], [4, 5]])[[0, 1, 2], [5, 5, 5]]

    (note,) = excinfo.value.__notes__
    assert "This error occurred while attempting to slice" in note
    assert "<Array [[1, 2, 3], [], [4, 5]] type='3 * var * int64'>" in note
    assert "([0, 1, 2], [5, 5, 5])" in note


def test_lazy_path_does_not_format_when_nothing_is_raised(monkeypatch):
    calls = []
    format_args = OperationErrorContext._format_args

    def spy(self, arguments):
        calls.append(arguments)
        return format_args(self, arguments)

    monkeypatch.setattr(OperationErrorContext, "_format_args", spy)

    array = ak.Array([[1, 2, 3], [], [4, 5]])
    assert ak.num(array, axis=1).to_list() == [3, 0, 2]
    assert calls == []

    with pytest.raises(AxisError):
        ak.num(array, axis=2)
    assert len(calls) == 1


@pytest.fixture
def delayed_cpu(monkeypatch):
    # Only a delayed nplike (CUDA) takes the eager path; `is_eager` is read
    # nowhere but `_errors`, so pretending the CPU nplike is delayed exercises
    # that path without a GPU.
    monkeypatch.setattr(Numpy, "is_eager", False)


def test_operation_context_formats_eagerly_for_delayed_backend(delayed_cpu):
    array = ak.Array([[1, 2, 3], [], [4, 5]])
    array_ref = weakref.ref(array)
    context = OperationErrorContext("ak.num", (array,), {"axis": 1})
    with context:
        pass

    assert context._args == ["<Array [[1, 2, 3], [], [4, 5]] type='3 * var * int64'>"]
    assert context._kwargs == {"axis": "1"}
    # the raw references must not be pinned by the (long-lived) context
    assert not hasattr(context, "_raw_args")
    assert not hasattr(context, "_raw_kwargs")
    del array
    gc.collect()
    assert array_ref() is None
    assert "ak.num(" in context.note


def test_slicing_context_formats_eagerly_for_delayed_backend(delayed_cpu):
    array = ak.Array([[1, 2, 3], [], [4, 5]])
    context = SlicingErrorContext(array, slice(1, None))
    with context:
        pass

    assert context._array == "<Array [[1, 2, 3], [], [4, 5]] type='3 * var * int64'>"
    assert context._where == "1:"
    assert not hasattr(context, "_raw_array")
    assert not hasattr(context, "_raw_where")


def test_eager_backend_contexts_stay_lazy():
    array = ak.Array([[1, 2, 3], [], [4, 5]])

    operation = OperationErrorContext("ak.num", (array,), {"axis": 1})
    slicing = SlicingErrorContext(array, slice(1, None))
    with operation:
        pass
    with slicing:
        pass

    assert operation._args is None
    assert operation._kwargs is None
    assert slicing._array is None
    assert slicing._where is None

    # ... but reading the note still formats them on demand
    assert "axis = 1" in operation.note
    assert "1:" in slicing.note
