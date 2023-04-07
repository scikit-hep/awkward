# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

__all__ = ("almost_equal",)


from awkward._backends import backend_of
from awkward._behavior import behavior_of, get_array_class, get_record_class
from awkward._nplikes.numpylike import NumpyMetadata
from awkward._parameters import parameters_are_equal
from awkward.operations.ak_to_layout import to_layout

np = NumpyMetadata.instance()


def almost_equal(
    left,
    right,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    dtype_exact: bool = True,
    check_parameters: bool = True,
    check_regular: bool = True,
) -> bool:
    """
    Args:
        left: Array-like data (anything #ak.to_layout recognizes).
        right: Array-like data (anything #ak.to_layout recognizes).
        rtol: the relative tolerance parameter (see below).
        atol: the absolute tolerance parameter (see below).
        dtype_exact: whether the dtypes must be exactly the same, or just the
            same family.
        check_parameters: whether to compare parameters.
        check_regular: whether to consider ragged and regular dimensions as
            unequal.

    Return True if the two array-like arguments are considered equal for the
    given options. Otherwise, return False.

    The relative difference (`rtol * abs(b)`) and the absolute difference `atol`
    are added together to compare against the absolute difference between `left`
    and `right`.

    TypeTracer arrays are not supported, as there is very little information to
    be compared.
    """
    left_behavior = behavior_of(left)
    right_behavior = behavior_of(right)

    left = to_layout(left, allow_record=False).to_packed()
    right = to_layout(right, allow_record=False).to_packed()

    backend = backend_of(left, right)
    if not backend.nplike.known_data:
        raise NotImplementedError(
            "Awkward Arrays with typetracer backends cannot yet be compared with `ak.almost_equal`."
        )

    def is_approx_dtype(left, right) -> bool:
        if not dtype_exact:
            for family in np.integer, np.floating:
                if np.issubdtype(left, family):
                    return np.issubdtype(right, family)
        return left == right

    def visitor(left, right) -> bool:
        # Enforce super-canonicalisation rules
        if left.is_option:
            left = left.to_IndexedOptionArray64()
        if right.is_option:
            right = right.to_IndexedOptionArray64()

        if type(left) is not type(right):
            if not check_regular and (
                left.is_list and right.is_regular or left.is_regular and right.is_list
            ):
                left = left.to_ListOffsetArray64()
                right = right.to_ListOffsetArray64()
            else:
                return False

        if left.length != right.length:
            return False

        if check_parameters and not parameters_are_equal(
            left.parameters, right.parameters
        ):
            return False

        # Require that the arrays have the same evaluated types
        if not (
            get_array_class(left, left_behavior)
            is get_array_class(right, right_behavior)
            or not check_parameters
        ):
            return False

        if left.is_list:
            return backend.index_nplike.array_equal(
                left.offsets, right.offsets
            ) and visitor(
                left.content[: left.offsets[-1]], right.content[: right.offsets[-1]]
            )
        elif left.is_regular:
            return (left.size == right.size) and visitor(left.content, right.content)
        elif left.is_numpy:
            return (
                is_approx_dtype(left.dtype, right.dtype)
                and backend.nplike.all(
                    backend.nplike.isclose(
                        left.data, right.data, rtol=rtol, atol=atol, equal_nan=False
                    )
                )
                and left.shape == right.shape
            )

        elif left.is_option:
            return backend.index_nplike.array_equal(
                left.index.data < 0, right.index.data < 0
            ) and visitor(left.project().to_packed(), right.project().to_packed())
        elif left.is_union:
            return (len(left.contents) == len(right.contents)) and all(
                visitor(left.project(i).to_packed(), right.project(i).to_packed())
                for i, _ in enumerate(left.contents)
            )
        elif left.is_record:
            return (
                (
                    get_record_class(left, left_behavior)
                    is get_record_class(right, right_behavior)
                    or not check_parameters
                )
                and (left.fields == right.fields)
                and (left.is_tuple == right.is_tuple)
                and all(visitor(x, y) for x, y in zip(left.contents, right.contents))
            )
        elif left.is_unknown:
            return True

        else:
            raise AssertionError

    return visitor(left, right)
