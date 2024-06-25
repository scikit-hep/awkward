# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest  # noqa: F401

import awkward as ak
from awkward.operations import to_list


def test_ak_where_with_optional_values():
    """
    This is the example from the Issue.
    In the two cases we fail, the not-selected value has type ?unknown, value None of course.
    """
    # Names here are changed a little from Github issue 3098.
    opt_true_cond = ak.Array([[True], [None]])[0]  # <Array [True] type='1 * ?bool'>
    true_cond = ak.Array([True])  # <Array [True] type='1 * bool'>
    none_alternative = ak.Array([None])  # <Array [None] type='1 * ?unknown'>
    zero_alternative = ak.Array([0])  # <Array [0] type='1 * int64'>
    opt_zero_alternative = ak.Array([[0], [None]])[0]  # <Array [0] type='1 * ?int64'>

    assert ak.where(opt_true_cond, 1, none_alternative).to_list() == [1]  # Fails at time of writing
    assert ak.where(opt_true_cond, 1, zero_alternative).to_list() == [1]
    assert ak.where(opt_true_cond, 1, opt_zero_alternative).to_list() == [1]

    # These assertions pass. Note that true_cond is type bool, not ?bool.
    assert ak.where(true_cond, 1, none_alternative).to_list() == [1]
    ak.where(true_cond, 1, zero_alternative).to_list() == [1]
    ak.where(true_cond, 1, opt_zero_alternative).to_list() == [1]

    # Like the first three assertions, The first one here fails.
    # This demonstrates that the problem is symmetric w/rt X and Y arrays.
    assert ak.where(~opt_true_cond, none_alternative, 1).to_list() == [1]  # Fails at time of writing
    assert ak.where(~opt_true_cond, zero_alternative, 1).to_list() == [1]
    assert ak.where(~opt_true_cond, opt_zero_alternative, 1).to_list() == [1]


def test_ak_with_no_optionals():
    """
    It turns out that we don't need to use ?unknown arrays to trigger this issue.
    We only need a None (masked value) in an element that is selected against.

    At time-of-writing (ATOW), ak.where() produces None values when:
    1. The conditional array values have OptionType (at least ?bool or ?int64), *AND*
    2. The value array element *NOT* selected has Option type and holds a None value.
    In this case regardless of the type or value of the array element that *IS* selected,
    the result for that element will, incorrectly, be None.
    """
    # This passes. Note that a condition of None creates a None in the result.
    assert to_list(
        ak.where(ak.Array([True, False, None]), ak.Array([1, 2, 3]), ak.Array([4, 5, 6]))
    ) == [1, 5, None]

    # This also passes. (The presence of None at the end forces option types to be used.)
    assert to_list(
        ak.where(ak.Array([True, False, None]), ak.Array([1, 2, None]), ak.Array([4, 5, None]))
    ) == [1, 5, None]

    # This fails (ATOW). The presence of None forces option types to be used.
    assert to_list(
        ak.where(ak.Array([True, False, None]), ak.Array([1, 2, None]), ak.Array([None, 5, None]))
    ) == [1, 5, None]  # ATOW we get [None, 5, None]

    # Fails ATOW. Same as above but with a None in the X argument.
    assert to_list(
        ak.where(ak.Array([True, False, None]), ak.Array([1, None, None]), ak.Array([4, 5, None]))
    ) == [1, 5, None]  # ATOW we get [1, None, None]

    # Fails ATOW. Same as above the Y argument is not even an optional type (but X still is).
    assert to_list(
        ak.where(ak.Array([True, False, None]), ak.Array([1, None, None]), ak.Array([4, 5, 6]))
    ) == [1, 5, None]  # ATOW we get [1, None, None]
