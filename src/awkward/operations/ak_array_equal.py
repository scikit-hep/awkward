# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function

__all__ = ("array_equal",)


@ak._connect.numpy.implements("array_equal")
@high_level_function()
def array_equal(
    a1,
    a2,
    equal_nan: bool = False,
    dtype_exact: bool = True,
    same_content_types: bool = True,
    check_parameters: bool = True,
    check_regular: bool = True,
    check_named_axis: bool = True,
):
    """
    Args:
        a1: Array-like data (anything #ak.to_layout recognizes).
        a2: Array-like data (anything #ak.to_layout recognizes).
        equal_nan: bool (default=False)
            Whether to count NaN values as equal to each other.
        dtype_exact: bool (default=True) whether the dtypes must be exactly the same, or just the
            same family.
        same_content_types: bool (default=True)
            Whether to require all content classes to match
        check_parameters: bool (default=True) whether to compare parameters.
        check_regular: bool (default=True) whether to consider ragged and regular dimensions as
            unequal.
        check_named_axis: bool (default=True) whether to consider named axes as unequal.

    True if two arrays have the same shape and elements, False otherwise.

    TypeTracer arrays are not supported, as there is very little information to
    be compared.
    """
    # Dispatch
    yield a1, a2

    return ak.operations.ak_almost_equal._impl(
        a1,
        a2,
        rtol=0.0,
        atol=0.0,
        dtype_exact=dtype_exact,
        check_parameters=check_parameters,
        check_regular=check_regular,
        check_named_axis=check_named_axis,
        exact_eq=True,
        same_content_types=same_content_types and check_regular,
        equal_nan=equal_nan,
    )
