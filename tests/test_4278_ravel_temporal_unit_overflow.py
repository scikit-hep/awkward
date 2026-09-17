# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak
from awkward._nplikes.numpy import Numpy
from awkward.contents.numpyarray import _check_temporal_merge_units

OPERATIONS = [
    pytest.param(ak.ravel, id="ravel"),
    pytest.param(lambda a: ak.flatten(a, axis=None), id="flatten-axis-None"),
]


def union(*arrays):
    return ak.concatenate([ak.Array(x) for x in arrays])


def record(**fields):
    return ak.Array(fields)


UNCONVERTIBLE = [
    pytest.param(
        union(np.array([1, 2], "m8[as]"), np.array(["NaT"], "m8[m]")),
        id="union-timedelta-as-m",
    ),
    pytest.param(
        record(x=np.array([1, 2], "m8[as]"), y=np.array([3, 4], "m8[m]")),
        id="record-timedelta-as-m",
    ),
    pytest.param(
        record(x=np.array([1, 2], "m8[fs]"), y=np.array([3, 4], "m8[h]")),
        id="record-timedelta-fs-h",
    ),
    pytest.param(
        record(x=np.array([1, 2], "M8[as]"), y=np.array([3, 4], "M8[s]")),
        id="record-datetime-as-s",
    ),
    pytest.param(
        record(x=np.array([1, 2], "M8[Y]"), y=np.array([3, 4], "M8[ps]")),
        id="record-datetime-calendar-Y-ps",
    ),
    pytest.param(
        record(
            x=np.array([1, 2], "m8[as]"),
            y=np.array([3, 4], "m8[ms]"),
            z=np.array([5, 6], "m8[s]"),
        ),
        id="record-timedelta-with-intermediate-unit",
    ),
    pytest.param(
        record(
            i=np.array([1, 2]),
            x=np.array([1, 2], "m8[as]"),
            y=np.array([3, 4], "m8[m]"),
        ),
        id="record-integer-beside-timedelta-pair",
    ),
    pytest.param(
        record(x=np.array([1, 2], "m8[100000as]"), y=np.array([3, 4], "m8[s]")),
        id="record-unit-multiplier-ignored",
    ),
    pytest.param(
        ak.Array(
            [
                [{"x": np.datetime64(1, "as"), "y": np.datetime64(2, "s")}],
                [],
                None,
            ]
        ),
        id="record-under-list-and-option",
    ),
    pytest.param(
        ak.Array([np.timedelta64(1, "as"), np.timedelta64(2, "m"), None]),
        id="option-of-union",
    ),
]

MERGEABLE = [
    pytest.param(
        record(x=np.array([1, 2], "m8[as]"), y=np.array([3, 4], "m8[ms]")),
        "4 * timedelta64[as]",
        id="record-timedelta-as-ms",
    ),
    pytest.param(
        record(x=np.array([1, 2], "M8[ns]"), y=np.array([3, 4], "M8[s]")),
        "4 * datetime64[ns]",
        id="record-datetime-ns-s",
    ),
    pytest.param(
        record(x=np.array([1, 2], "m8[as]"), y=np.array([3, 4], "m8[1000000ms]")),
        "4 * timedelta64[as]",
        id="record-unit-multiplier-under-bound",
    ),
    pytest.param(
        record(i=np.array([1, 2]), y=np.array([3, 4], "m8[s]")),
        "4 * timedelta64[s]",
        id="record-integer-beside-timedelta",
    ),
    pytest.param(
        record(x=np.array([1, 2], "m8[as]"), y=np.array([3, 4], "m8[as]")),
        "4 * timedelta64[as]",
        id="record-one-unit",
    ),
    pytest.param(
        union(np.array([1, 2], "m8[as]"), np.array([3, 4], "m8[ms]")),
        "4 * timedelta64[as]",
        id="union-timedelta-as-ms",
    ),
    pytest.param(
        record(x=np.array([1, 2], "m8[us]"), y=np.array([3, 4], "m8[ps]")),
        "4 * timedelta64[ps]",
        id="record-timedelta-us-ps",
    ),
]

NOT_A_UNIT_CONVERSION = [
    pytest.param(
        record(x=np.array([1, 2], "M8[as]"), y=np.array([3, 4], "m8[s]")),
        OverflowError,
        id="datetime-with-timedelta-far-apart-units",
    ),
    pytest.param(
        record(x=np.array([1, 2], "m8[Y]"), y=np.array([3, 4], "m8[s]")),
        TypeError,
        id="calendar-with-linear-timedelta",
    ),
    pytest.param(
        record(x=np.array([1.5, 2.5]), y=np.array([3, 4], "m8[s]")),
        TypeError,
        id="float-with-timedelta",
    ),
]


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("array", UNCONVERTIBLE)
def test_unconvertible_units_raise_value_error(operation, array):
    with pytest.raises(ValueError, match="cannot merge"):
        operation(array)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_error_names_the_temporal_dtypes(operation):
    array = record(
        i=np.array([1, 2]),
        x=np.array([1, 2], "m8[as]"),
        y=np.array([3, 4], "m8[m]"),
    )
    with pytest.raises(
        ValueError, match=r"cannot merge timedelta64\[as\], timedelta64\[m\]:"
    ):
        operation(array)


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize(("array", "expected_type"), MERGEABLE)
def test_convertible_units_still_merge(operation, array, expected_type):
    assert str(operation(array).type) == expected_type


@pytest.mark.parametrize("operation", OPERATIONS)
def test_unconvertible_units_raise_on_typetracer(operation):
    array = record(x=np.array([1, 2], "m8[as]"), y=np.array([3, 4], "m8[m]"))
    with pytest.raises(ValueError, match="cannot merge"):
        operation(ak.Array(array.layout.to_typetracer(forget_length=True)))


def test_unreached_leaf_does_not_force_a_merge():
    array = union(np.array([], "m8[as]"), np.array(["NaT"], "m8[m]"))
    assert str(ak.ravel(array).type) == "1 * timedelta64[m]"


def test_concatenate_still_builds_a_union():
    array = union(np.array([1, 2], "m8[as]"), np.array([3, 4], "m8[ms]"))
    assert str(array.type) == "4 * union[timedelta64[as], timedelta64[ms]]"


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize(("array", "expected_error"), NOT_A_UNIT_CONVERSION)
def test_other_merge_failures_are_unchanged(operation, array, expected_error):
    with pytest.raises(expected_error):
        operation(array)


def test_the_units_are_judged_without_the_values():
    nplike = Numpy.instance()
    _check_temporal_merge_units([np.empty(0, "m8[us]"), np.empty(0, "m8[ps]")], nplike)
    with pytest.raises(ValueError, match="cannot merge"):
        _check_temporal_merge_units(
            [np.empty(0, "m8[as]"), np.empty(0, "m8[s]")], nplike
        )
