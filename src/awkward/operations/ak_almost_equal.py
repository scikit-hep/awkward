# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._backends.dispatch import backend_of
from awkward._backends.numpy import NumpyBackend
from awkward._behavior import behavior_of, get_array_class, get_record_class
from awkward._dispatch import high_level_function
from awkward._layout import ensure_same_backend
from awkward._namedaxis import _get_named_axis
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._parameters import parameters_are_equal
from awkward.operations.ak_to_layout import to_layout

__all__ = ("almost_equal",)

np = NumpyMetadata.instance()
cpu = NumpyBackend.instance()


@high_level_function()
def almost_equal(
    left,
    right,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    dtype_exact: bool = True,
    check_parameters: bool = True,
    check_regular: bool = True,
    check_named_axis: bool = True,
):
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
        check_named_axis: bool (default=True) whether to consider named axes as unequal.

    Return True if the two array-like arguments are considered equal for the
    given options. Otherwise, return False.

    The relative difference (`rtol * abs(b)`) and the absolute difference `atol`
    are added together to compare against the absolute difference between `left`
    and `right`.

    TypeTracer arrays are not supported, as there is very little information to
    be compared.
    """
    # Dispatch
    yield left, right

    return _impl(
        left,
        right,
        rtol=rtol,
        atol=atol,
        dtype_exact=dtype_exact,
        check_parameters=check_parameters,
        check_regular=check_regular,
        check_named_axis=check_named_axis,
        exact_eq=False,
        same_content_types=False,
        equal_nan=False,
    )


def _impl(
    left,
    right,
    rtol: float,
    atol: float,
    dtype_exact: bool,
    check_parameters: bool,
    check_regular: bool,
    check_named_axis: bool,
    exact_eq: bool,
    same_content_types: bool,
    equal_nan: bool,
):
    # Implementation
    left_behavior = behavior_of(left)
    right_behavior = behavior_of(right)

    layouts = ensure_same_backend(
        to_layout(left, allow_record=False),
        to_layout(right, allow_record=False),
    )
    left_layout = layouts[0].to_packed()
    right_layout = layouts[1].to_packed()
    backend = backend_of(left_layout)

    if check_named_axis and _get_named_axis(left) and _get_named_axis(right):
        if left.named_axis != right.named_axis:
            return False

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

    def packed_list_content(layout):
        layout = layout.to_ListOffsetArray64(False)
        return layout.content[layout.offsets[0] : layout.offsets[-1]]

    def visitor(left, right) -> bool:
        # Most firstly, check same_content_types before any transformations
        if same_content_types and left.__class__ is not right.__class__:
            return False

        # First, erase indexed types!
        if left.is_indexed and not left.is_option:
            left = left.project()
        if right.is_indexed and not right.is_option:
            right = right.project()

        # Simplify option types
        if left.is_option:
            left = left.to_IndexedOptionArray64()
        if right.is_option:
            right = right.to_IndexedOptionArray64()

        # Simplify regular NumPy types
        if left.is_numpy and left.purelist_depth > 1:
            left = left.to_RegularArray()
        if right.is_numpy and right.purelist_depth > 1:
            right = right.to_RegularArray()

        # Different lengths aren't equal!
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

        # Regular-regular
        if left.is_regular and right.is_regular:
            return (left.size == right.size) and visitor(left.content, right.content)
        # List-list
        elif left.is_list and right.is_list:
            # Check that indexes are equal
            left_index = left.to_ListOffsetArray64(True).offsets
            right_index = right.to_ListOffsetArray64(True).offsets
            if not backend.nplike.array_equal(left_index.data, right_index.data):
                return False
            # Mixed regular-var
            if left.is_regular and not right.is_regular:
                return (not check_regular) and visitor(
                    left.content,
                    packed_list_content(right),
                )
            elif right.is_regular and not left.is_regular:
                return (not check_regular) and visitor(
                    packed_list_content(left),
                    right.content,
                )
            else:
                return visitor(
                    packed_list_content(left),
                    packed_list_content(right),
                )

        elif left.is_numpy and right.is_numpy:
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
            elif exact_eq:
                return (
                    is_approx_dtype(left.dtype, right.dtype)
                    and backend.nplike.array_equal(
                        left.data,
                        right.data,
                        equal_nan=equal_nan,
                    )
                    and left.shape == right.shape
                )
            else:
                return (
                    is_approx_dtype(left.dtype, right.dtype)
                    and backend.nplike.all(
                        backend.nplike.isclose(
                            left.data,
                            right.data,
                            rtol=rtol,
                            atol=atol,
                            equal_nan=equal_nan,
                        )
                    )
                    and left.shape == right.shape
                )
        elif left.is_option and right.is_option:
            return backend.nplike.array_equal(
                left.mask_as_bool(True), right.mask_as_bool(True)
            ) and visitor(left.project(), right.project())
        elif left.is_union and right.is_union:
            # Group positions by (left_tag, right_tag) pairs
            # This allows comparing elements in bulk rather than one-by-one
            tag_pair_to_positions = {}
            for i in range(left.length):
                pair = (int(left.tags[i]), int(right.tags[i]))
                if pair not in tag_pair_to_positions:
                    tag_pair_to_positions[pair] = []
                tag_pair_to_positions[pair].append(i)

            # For each unique tag pair, extract and compare elements in bulk
            for (left_tag, right_tag), positions in tag_pair_to_positions.items():
                positions_array = backend.nplike.asarray(positions, dtype=np.int64)

                # Get the indices into the content arrays
                left_indices = left.index.data[positions_array]
                right_indices = right.index.data[positions_array]

                # Extract and compare the elements
                if not visitor(
                    left.content(left_tag)[left_indices],
                    right.content(right_tag)[right_indices],
                ):
                    return False

            return True

        elif left.is_record and right.is_record:
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
        elif left.is_unknown and right.is_unknown:
            return True

        else:
            return False

    return visitor(left_layout, right_layout)
