# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

# ak.ravel, ak.flatten(axis=None), and axis=None reductions merge the leaves they
# collect, and merging temporal leaves of one family converts every value to the
# unit the merge takes. A value of magnitude above (2**63 - 1) // factor, for that
# leaf's conversion factor, has no int64 representation there: NumPy 2.5.0 and later
# raise a bare OverflowError from the cast, and earlier versions wrap the value
# around silently. Awkward now checks the range itself and raises ValueError on
# every supported NumPy version.

# The conversion factor from [us] to [ps] is 10**6.
US_PS_LIMIT = (2**63 - 1) // 10**6
# ... and from [ms] to [us], 10**3.
MS_US_LIMIT = (2**63 - 1) // 10**3

MESSAGE = "cannot merge timedelta64|cannot merge datetime64"


def union_of(first, second):
    return ak.concatenate([ak.Array(first), ak.Array(second)])


def record_of(first, second):
    return ak.Array({"x": first, "y": second})


@pytest.mark.parametrize("layout_of", [union_of, record_of])
@pytest.mark.parametrize("kind", ["m8", "M8"])
def test_ravel_value_above_merged_unit_raises(layout_of, kind):
    array = layout_of(
        np.array([US_PS_LIMIT + 1], f"{kind}[us]"), np.array([0], f"{kind}[ps]")
    )
    with pytest.raises(ValueError, match=MESSAGE):
        ak.ravel(array)
    with pytest.raises(ValueError, match=MESSAGE):
        ak.flatten(array, axis=None)


@pytest.mark.parametrize("layout_of", [union_of, record_of])
@pytest.mark.parametrize("kind", ["m8", "M8"])
def test_ravel_value_at_merged_unit_bound_succeeds(layout_of, kind):
    array = layout_of(
        np.array([US_PS_LIMIT], f"{kind}[us]"), np.array([0], f"{kind}[ps]")
    )
    expected = [US_PS_LIMIT * 10**6, 0]
    assert ak.ravel(array).to_numpy().view(np.int64).tolist() == expected
    assert ak.flatten(array, axis=None).to_numpy().view(np.int64).tolist() == expected


@pytest.mark.parametrize("kind", ["m8", "M8"])
def test_ravel_negative_bound_is_symmetric(kind):
    at_bound = union_of(
        np.array([-US_PS_LIMIT], f"{kind}[us]"), np.array([0], f"{kind}[ps]")
    )
    assert ak.ravel(at_bound).to_numpy().view(np.int64).tolist() == [
        -US_PS_LIMIT * 10**6,
        0,
    ]

    beyond = union_of(
        np.array([-US_PS_LIMIT - 1], f"{kind}[us]"), np.array([0], f"{kind}[ps]")
    )
    with pytest.raises(ValueError, match=MESSAGE):
        ak.ravel(beyond)


@pytest.mark.parametrize("layout_of", [union_of, record_of])
@pytest.mark.parametrize("kind", ["m8", "M8"])
def test_not_a_time_alone_converts_to_the_finer_unit(layout_of, kind):
    # The [us] leaf holds nothing but NaT and merges to the finer [ps]. No
    # conversion factor applies to NaT, so it survives the merge whatever
    # the factor is.
    array = layout_of(np.array(["NaT"], f"{kind}[us]"), np.array([0], f"{kind}[ps]"))
    merged = ak.ravel(array).to_numpy()
    assert np.isnat(merged).tolist() == [True, False]
    assert merged[~np.isnat(merged)].view(np.int64).tolist() == [0]


@pytest.mark.parametrize("layout_of", [union_of, record_of])
@pytest.mark.parametrize("kind", ["m8", "M8"])
def test_not_a_time_beside_values_at_the_bounds(layout_of, kind):
    # NaT is int64's minimum, so a scan of the raw buffer would read it as
    # far below the lower bound. It must not be taken for an out-of-range
    # value, and the two bounds sharing its leaf must still merge.
    array = layout_of(
        np.array(["NaT", US_PS_LIMIT, -US_PS_LIMIT], f"{kind}[us]"),
        np.array([0, 0, 0], f"{kind}[ps]"),
    )
    merged = ak.ravel(array).to_numpy()
    assert np.isnat(merged).sum() == 1
    assert sorted(merged[~np.isnat(merged)].view(np.int64).tolist()) == sorted(
        [US_PS_LIMIT * 10**6, -US_PS_LIMIT * 10**6, 0, 0, 0]
    )


def test_ravel_milliseconds_with_microseconds():
    at_bound = record_of(np.array([MS_US_LIMIT], "m8[ms]"), np.array([0], "m8[us]"))
    assert ak.ravel(at_bound).to_numpy().view(np.int64).tolist() == [
        MS_US_LIMIT * 10**3,
        0,
    ]

    beyond = record_of(np.array([MS_US_LIMIT + 1], "m8[ms]"), np.array([0], "m8[us]"))
    with pytest.raises(ValueError, match=MESSAGE):
        ak.ravel(beyond)


def test_ravel_unit_multiplier_enters_the_factor():
    # A step of 10 [us] converts to [ps] by 10**7, not 10**6.
    limit = (2**63 - 1) // 10**7
    at_bound = union_of(np.array([limit], "m8[10us]"), np.array([0], "m8[ps]"))
    assert ak.ravel(at_bound).to_numpy().view(np.int64).tolist() == [
        limit * 10**7,
        0,
    ]

    beyond = union_of(np.array([limit + 1], "m8[10us]"), np.array([0], "m8[ps]"))
    with pytest.raises(ValueError) as excinfo:
        ak.ravel(beyond)
    message = str(excinfo.value)
    assert "cannot merge timedelta64[10us] with timedelta64[ps]" in message
    assert f"cannot represent {limit + 1}" in message
    assert f"from {-limit} to {limit} fit" in message


def test_error_reports_plain_integers_for_a_multiplier_dtype():
    # Printing a temporal scalar scales it by the dtype's unit multiplier,
    # which overflows at these magnitudes: `m8[6s]` renders the bound as
    # "-4 seconds" and the value as "-9223372036854775808 seconds". The
    # message carries the raw integers instead, in units of the dtype
    # named beside them.
    value = 2**62
    limit = (2**63 - 1) // 3  # m8[6s] merges with m8[4s] to m8[2s]
    array = record_of(np.array([value], "m8[6s]"), np.array([0], "m8[4s]"))
    with pytest.raises(ValueError) as excinfo:
        ak.ravel(array)
    message = str(excinfo.value)
    assert "cannot merge timedelta64[6s] with timedelta64[2s]" in message
    assert f"cannot represent {value}" in message
    assert f"from {-limit} to {limit} fit" in message
    # No value is rendered through the dtype, so no unit name appears.
    assert "seconds" not in message


def test_bound_is_exact_when_the_merged_step_does_not_divide_the_step():
    # NumPy merges these to timedelta64[13ns], and one 100000-week step is
    # not a whole number of 13-nanosecond steps (6.048e28 attoseconds
    # against 1.3e10). The bound is computed without forming that ratio,
    # so truncating it cannot loosen the bound: only magnitudes up to 1 fit.
    coarse, fine = "m8[100000W]", "m8[13ns]"
    for value in (0, 1, -1):
        ak.ravel(record_of(np.array([value], coarse), np.array([0], fine)))
    for value in (2, -2):
        with pytest.raises(ValueError, match="from -1 to 1 fit"):
            ak.ravel(record_of(np.array([value], coarse), np.array([0], fine)))


def test_neighbouring_merge_failures_are_unchanged():
    # The range check hands back every pair whose merged dtype it cannot
    # determine, so the failures tracked in #4278 and #4261 keep the errors
    # they had. The known-issue predicates of the property tests rely on it.

    # #4278: a unit factor NumPy refuses, before any value is converted.
    with pytest.raises(OverflowError):
        ak.flatten(
            union_of(np.array([], "m8[as]"), np.array(["NaT"], "m8[m]")), axis=None
        )

    # #4261: leaves of families that do not promote.
    with pytest.raises(AssertionError):
        ak.flatten(ak.Array([{"x": 1, "y": "s"}]), axis=None)
    with pytest.raises(AttributeError):
        ak.flatten(ak.Array([{"x": "s", "y": 1}]), axis=None)
    # NumPy's DTypePromotionError, a TypeError on every supported version.
    with pytest.raises(TypeError):
        ak.ravel(record_of(np.array([1.0]), np.array([1], "m8[ms]")))
    # A nonlinear unit pair, which has no conversion factor at all.
    with pytest.raises(TypeError):
        ak.ravel(record_of(np.array([1], "m8[ms]"), np.array([1], "m8[M]")))


def test_reductions_with_axis_none_raise():
    array = union_of(np.array([US_PS_LIMIT + 1], "m8[us]"), np.array([0], "m8[ps]"))
    with pytest.raises(ValueError, match=MESSAGE):
        ak.sum(array, axis=None)
    with pytest.raises(ValueError, match=MESSAGE):
        ak.all(array, axis=None)


def test_error_names_the_units_and_the_value():
    array = union_of(np.array([US_PS_LIMIT + 1], "m8[us]"), np.array([0], "m8[ps]"))
    with pytest.raises(ValueError) as excinfo:
        ak.ravel(array)
    message = str(excinfo.value)
    assert "timedelta64[us]" in message
    assert "timedelta64[ps]" in message
    assert f"cannot represent {US_PS_LIMIT + 1}" in message
    assert f"from {-US_PS_LIMIT} to {US_PS_LIMIT} fit" in message
    # The operation is named by the error context, which attaches a note.
    assert any("ak.ravel(" in note for note in excinfo.value.__notes__)


def test_a_third_leaf_does_not_hide_the_overflow():
    array = ak.Array(
        {
            "x": np.array([1]),
            "y": np.array([US_PS_LIMIT + 1], "m8[us]"),
            "z": np.array([0], "m8[ps]"),
        }
    )
    with pytest.raises(ValueError, match=MESSAGE):
        ak.ravel(array)


def test_values_astype_is_a_way_through():
    array = union_of(np.array([US_PS_LIMIT + 1], "m8[us]"), np.array([0], "m8[ps]"))
    cast = ak.values_astype(array, "timedelta64[us]")
    assert ak.ravel(cast).to_numpy().view(np.int64).tolist() == [US_PS_LIMIT + 1, 0]


def test_merges_that_already_worked_are_unchanged():
    same_unit = record_of(np.array([5], "m8[us]"), np.array([6], "m8[us]"))
    assert ak.ravel(same_unit).to_numpy().view(np.int64).tolist() == [5, 6]

    small_values = record_of(np.array([5], "m8[us]"), np.array([6], "m8[ns]"))
    assert ak.ravel(small_values).to_numpy().view(np.int64).tolist() == [5000, 6]

    with_integer = record_of(np.array([5]), np.array([6], "m8[us]"))
    assert ak.ravel(with_integer).to_numpy().view(np.int64).tolist() == [5, 6]

    # ak.concatenate keeps its union fallback, so it never reaches the merge.
    union = union_of(np.array([US_PS_LIMIT + 1], "m8[us]"), np.array([0], "m8[ps]"))
    assert str(union.type) == "2 * union[timedelta64[us], timedelta64[ps]]"


def test_typetracer_cannot_decide_and_does_not_raise():
    array = union_of(np.array([US_PS_LIMIT + 1], "m8[us]"), np.array([0], "m8[ps]"))
    typetracer = ak.Array(array.layout.to_typetracer(forget_length=True))
    assert str(ak.ravel(typetracer).type) == "## * timedelta64[ps]"
