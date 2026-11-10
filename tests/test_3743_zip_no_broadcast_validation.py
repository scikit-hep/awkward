# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.placeholder import PlaceholderArray
from awkward.operations.ak_zip_no_broadcast import _check_equal_lengths


@pytest.mark.parametrize("arrays", [{}, [], ()])
def test_empty_inputs(arrays):
    with pytest.raises(ValueError, match="requires at least one array"):
        ak.zip_no_broadcast(arrays)


@pytest.mark.parametrize("arrays", [[1], [None], [1, [2]], [[2], 1], [None, [2]]])
def test_non_array_inputs(arrays):
    with pytest.raises(ValueError, match="all array layouts must be either"):
        ak.zip_no_broadcast(arrays)


@pytest.mark.parametrize(
    "contents", [[["a"], []], [[[1]], []], [[None], []], [[{"x": 1}], []]]
)
def test_unsupported_list_contents(contents):
    with pytest.raises(NotImplementedError, match=r"Use ak\.zip instead"):
        ak.zip_no_broadcast({"x": contents})


def test_issue_reproducer():
    one = [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6]]
    two = [["a", "b", "c"], [], ["d", "e"], ["f"]]
    with pytest.raises(NotImplementedError, match="contents are not NumpyArray"):
        ak.zip_no_broadcast({"x": one, "y": two})
    assert ak.zip({"x": one, "y": two}).to_list()[0][0] == {"x": 1.1, "y": "a"}


@pytest.mark.parametrize(
    "parameters", [None, {"custom": "preserved", "__record__": "Old"}]
)
def test_record_metadata(parameters):
    first = ak.with_named_axis(ak.Array([1, 2]), {"row": 0})
    second = ak.with_named_axis(ak.Array([3, 4]), {"row": 0})
    original = None if parameters is None else parameters.copy()
    behavior = {}
    result = ak.zip_no_broadcast(
        {"x": first, "y": second},
        parameters=parameters,
        with_name="Pair",
        behavior=behavior,
        attrs={"source": "test"},
    )
    assert result.to_list() == [{"x": 1, "y": 3}, {"x": 2, "y": 4}]
    assert result.layout.parameter("__record__") == "Pair"
    assert result.named_axis == {"row": 0}
    assert result.attrs["source"] == "test"
    assert result.behavior is behavior
    assert parameters == original
    if parameters is not None:
        assert result.layout.parameter("custom") == "preserved"


def test_tuple_lowlevel_and_iterator():
    result = ak.zip_no_broadcast(iter(([1, 2], [3, 4])), highlevel=False)
    assert isinstance(result, ak.contents.RecordArray)
    assert result.is_tuple
    assert ak.to_list(result) == [(1, 3), (2, 4)]


def test_unequal_numeric_lengths():
    with pytest.raises(ValueError, match="all arrays must have the same length"):
        ak.zip_no_broadcast([[1], [2, 3]])


@pytest.mark.parametrize("second", [[[3], [4, 5]], [[3, 4, 5]], [[3], [], [4, 5]]])
def test_unequal_offsets(second):
    with pytest.raises(ValueError, match="same offsets"):
        ak.zip_no_broadcast([[[1, 2], [3]], second])


def test_unequal_content_lengths():
    offsets = ak.index.Index64(np.array([0, 1], dtype=np.int64))
    first = ak.contents.ListOffsetArray(offsets, ak.contents.NumpyArray(np.array([1])))
    second = ak.contents.ListOffsetArray(
        offsets, ak.contents.NumpyArray(np.array([2, 3]))
    )
    with pytest.raises(ValueError, match="same length"):
        ak.zip_no_broadcast([first, second])


def test_mixed_layouts():
    with pytest.raises(ValueError, match="all array layouts must be either"):
        ak.zip_no_broadcast([[1, 2], [[3], [4]]])


@pytest.mark.parametrize("concrete", [False, True])
def test_placeholder_offsets(concrete):
    placeholder = PlaceholderArray(Numpy.instance(), (3,), np.dtype(np.int64))
    first = ak.contents.ListOffsetArray(
        ak.index.Index64(placeholder), ak.contents.NumpyArray(np.array([1, 2, 3]))
    )
    second = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 2, 3], dtype=np.int64))
        if concrete
        else first.offsets,
        ak.contents.NumpyArray(np.array([4, 5, 6])),
    )
    result = ak.zip_no_broadcast([first, second], highlevel=False)
    assert result.offsets is (second.offsets if concrete else first.offsets)
    assert result.content.to_list() == [(1, 4), (2, 5), (3, 6)]


def test_empty_length_helper():
    with pytest.raises(ValueError, match="requires at least one content"):
        _check_equal_lengths([])
