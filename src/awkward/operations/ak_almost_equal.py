# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

__all__ = ("almost_equal",)


from awkward._backends.dispatch import backend_of
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
            # Timelike types must be exactly compared, including their units
            if (
                np.issubdtype(left.dtype, np.datetime64)
                or np.issubdtype(right.dtype, np.datetime64)
                or np.issubdtype(left.dtype, np.timedelta64)
                or np.issubdtype(right.dtype, np.timedelta64)
            ):
                return (
                    (left.dtype == right.dtype)
                    and backend.nplike.all(left.data == right.data)
                    and left.shape == right.shape
                )
            else:
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
            # For two unions with different content orderings to match, the tags should be equal at each index
            # Therefore, we can order the contents by index appearance
            def ordered_unique_values(values):
                # First, find unique values and their appearance (from smallest to largest)
                # unique_index is in ascending order of `unique` value
                (
                    unique,
                    unique_index,
                    *_,
                ) = backend.index_nplike.unique_all(values)
                # Now re-order `unique` by order of appearance (`unique_index`)
                return values[backend.index_nplike.sort(unique_index)]

            # Find order of appearance for each union tags, and assume these are one-to-one maps
            left_tag_order = ordered_unique_values(left.tags.data)
            right_tag_order = ordered_unique_values(right.tags.data)

            # Create map from left tags to right tags
            left_tag_to_right_tag = backend.index_nplike.empty(
                left_tag_order.size, dtype=np.int64
            )
            left_tag_to_right_tag[left_tag_order] = right_tag_order

            # Map left tags onto right, such that the result should equal right.tags
            # if the two tag arrays are equivalent
            new_left_tag = left_tag_to_right_tag[left.tags.data]
            if not backend.index_nplike.all(new_left_tag == right.tags.data):
                return False

            # Now project out the contents, and check for equality
            for i, j in zip(left_tag_order, right_tag_order):
                if not visitor(
                    left.project(i).to_packed(), right.project(j).to_packed()
                ):
                    return False
            return True

        elif left.is_record:
            return (
                (
                    get_record_class(left, left_behavior)
                    is get_record_class(right, right_behavior)
                    or not check_parameters
                )
                and left.is_tuple == right.is_tuple
                and (left.is_tuple or (len(left.fields) == len(right.fields)))
                and all(visitor(left.content(f), right.content(f)) for f in left.fields)
            )
        elif left.is_unknown:
            return True

        else:
            raise AssertionError

    return visitor(left, right)
