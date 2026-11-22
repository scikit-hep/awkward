# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import operator

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._experimental import experimental
from awkward._layout import HighLevelContext, maybe_posaxis
from awkward._namedaxis import _get_named_axis, _named_axis_to_positional_axis
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._nplikes.shape import unknown_length
from awkward._regularize import is_integer, regularize_axis
from awkward._typing import Any, NamedTuple
from awkward.errors import AxisError

__all__ = ("top_k",)

np = NumpyMetadata.instance()


class TopKResult(NamedTuple):
    values: Any
    indices: Any


@high_level_function()
@experimental
def top_k(
    array, k, /, *, axis=-1, mode="largest", highlevel=True, behavior=None, attrs=None
):
    """Return the largest or smallest values and their indices along an axis.

    Args:
        array: Array-like input with real numeric or boolean values.
        k (int): Nonnegative number of entries to select from each list.
        axis (int or str): Axis to search; defaults to the innermost axis, -1.
            Named axes are supported. None is not accepted.
        mode (str): Either "largest" (the default) or "smallest".
        highlevel (bool): Return high-level arrays if True, otherwise layouts.
        behavior (None or dict): Custom behavior for both output arrays.
        attrs (None or dict): Custom attributes for both output arrays.

    Returns:
        A namedtuple with fields `values` and `indices`. The values retain the
        input dtype; indices refer to original positions along the searched
        axis, including positions occupied by missing rows.

    This follows the Array API `top_k` signature and namedtuple result, with
    Awkward's usual output options. Lists shorter than `k` return all their
    entries. Results are sorted from largest to smallest in "largest" mode,
    or smallest to largest in "smallest" mode. Equal values retain input order.
    Missing values occur after present values; they are retained if the
    selected range reaches them. Floating-point values are ranked by IEEE 754
    total order with NaN as the largest value, as in NumPy: NaNs come first in
    "largest" mode and last in "smallest" mode, and -0.0 ranks below 0.0. NaNs
    are returned with their sign and payload normalized.

    Regular dimensions of the input remain regular in the output, with the
    searched dimension having size `k` (or its original size, if smaller).

    For ragged outer-axis selection, entries at matching inner coordinates are
    compared, as in Awkward reductions. The result retains each group's union of
    inner coordinates, with None padding where a column has fewer entries than
    its neighbors. Selected missing values have None indices. For example,
    ``ak.top_k([[3], [2, 4], [0, 1]], 1, axis=0).values`` is ``[[3, 4]]``.
    Alignment can require temporary padding of inner dimensions to their longest
    length; it may use substantially more memory than the original ragged array.
    On GPU, discovering these padding widths requires scalar synchronization.

    This initial implementation performs a full sort, not a partial selection.
    #ak.top_k_values and #ak.top_k_indices return only the requested output.

        >>> result = ak.top_k([[3, 1, 5], [], [2, 4]], 2)
        >>> result.values.to_list()
        [[5, 3], [], [4, 2]]
        >>> result.indices.to_list()
        [[2, 0], [], [1, 0]]
    """
    yield (array,)
    return _impl(array, k, axis, mode, highlevel, behavior, attrs, "both")


def _impl(array, k, axis, mode, highlevel, behavior, attrs, output):
    if not is_integer(k):
        raise TypeError("k must be a nonnegative integer")
    k = operator.index(k)
    if k < 0:
        raise ValueError("k must be a nonnegative integer")
    if mode not in ("largest", "smallest"):
        raise ValueError("mode must be 'largest' or 'smallest'")

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False, primitive_policy="error")
    axis = regularize_axis(
        _named_axis_to_positional_axis(_get_named_axis(ctx), axis), none_allowed=False
    )
    posaxis = maybe_posaxis(layout, axis, 1)
    if posaxis is None:
        raise ValueError("top_k requires an unambiguous axis")
    if posaxis < 0 or posaxis >= layout.minmax_depth[0]:
        raise AxisError(f"axis={axis} exceeds the depth of the input array")

    needs_alignment = False
    has_option = False
    regular_sizes = {}
    leaf_dtypes = set()

    def validate(node, depth, **kwargs):
        nonlocal needs_alignment, has_option
        has_option = has_option or node.is_option
        if node.is_numpy:
            leaf_dtypes.add(node.dtype)
        if node.is_list:
            size = node.size if node.is_regular else None
            regular_sizes[depth] = (
                size if regular_sizes.get(depth, size) == size else None
            )
        if node.is_record or node.parameter("__array__") in ("string", "bytestring"):
            raise TypeError("top_k requires real numeric or boolean values")
        if node.is_numpy and node.dtype.kind not in "biuf":
            raise TypeError("top_k requires real numeric or boolean values")
        if depth > posaxis and (
            (node.is_list and not node.is_regular)
            or (node.is_option and node.minmax_depth[1] > 1)
        ):
            # Missing list rows must retain their original axis positions,
            # including when their content has a regular inner dimension.
            needs_alignment = True

    ak._do.recursively_apply(layout, validate, return_array=False)
    counts = {}
    widths = {}
    if needs_alignment:
        layout, counts, widths = _align(layout, posaxis)
    ascending = mode == "smallest"
    # Without records or unions, there is at most one numeric leaf.
    float_dtype = next(iter(leaf_dtypes)) if len(leaf_dtypes) == 1 else None
    if float_dtype is not None and _float_key_dtype(float_dtype) is None:
        float_dtype = None
    if float_dtype is not None:
        layout = _floats_to_keys(layout)
    values = None
    if output != "indices" or has_option or needs_alignment:
        values = _select(layout, posaxis, k, ascending, indices=False)
    indices = None
    if output != "values":
        indices = _select(layout, posaxis, k, ascending, indices=True)
        if has_option or needs_alignment:
            # Strip parameters so that `~` does not dispatch to behaviors.
            missing = ak.without_parameters(ak.is_none(values, axis=-1))
            indices = ak.mask(indices, ~missing, highlevel=False)
    if values is not None and float_dtype is not None:
        values = _keys_to_floats(values, float_dtype)
    if needs_alignment:
        values = _trim_alignment(values, counts, widths, posaxis)
        if indices is not None:
            indices = _trim_alignment(indices, counts, widths, posaxis)
    else:
        # Sorting returns variable-length lists; restore regular dimensions.
        sizes = {
            depth: min(k, size) if depth == posaxis else size
            for depth, size in regular_sizes.items()
            if size is not None
        }
        values = _restore_regular(values, sizes)
        indices = _restore_regular(indices, sizes)
    if output == "values":
        return ctx.wrap(values, highlevel=highlevel)
    if output == "indices":
        return ctx.wrap(indices, highlevel=highlevel)
    return TopKResult(
        ctx.wrap(values, highlevel=highlevel), ctx.wrap(indices, highlevel=highlevel)
    )


def _float_key_dtype(dtype):
    if dtype.kind == "f" and dtype.itemsize in (2, 4, 8):
        return np.dtype(f"i{dtype.itemsize}")
    return None


def _flip_negative(nplike, bits):
    # Involution mapping IEEE 754 bit patterns (viewed as signed integers) to
    # integers with the same total order as the floats, and back again.
    int_min = np.iinfo(bits.dtype).min
    return nplike.where(bits < 0, int_min - (bits + 1), bits)


def _floats_to_keys(layout):
    # The sort kernels place NaN first in both directions. Sorting integer
    # keys instead ranks NaN as the largest value in both modes.
    def action(node, **kwargs):
        if node.is_numpy and _float_key_dtype(node.dtype) is not None:
            nplike = node.backend.nplike
            data = node.data
            data = nplike.where(
                nplike.isnan(data), nplike.full_like(data, np.nan), data
            )
            return ak.contents.NumpyArray(
                _flip_negative(nplike, data.view(_float_key_dtype(node.dtype))),
                parameters=node.parameters,
                backend=node.backend,
            )
        return None

    return ak._do.recursively_apply(layout, action)


def _keys_to_floats(layout, dtype):
    def action(node, **kwargs):
        if node.is_numpy:
            return ak.contents.NumpyArray(
                _flip_negative(node.backend.nplike, node.data).view(dtype),
                parameters=node.parameters,
                backend=node.backend,
            )
        return None

    return ak._do.recursively_apply(layout, action)


def _restore_regular(layout, sizes):
    if layout is None:
        return None
    for target in sorted(sizes, reverse=True):
        size = sizes[target]

        def action(node, depth, target=target, size=size, **kwargs):
            if node.is_list and depth == target:
                if node.is_regular:
                    return node
                node = node.to_ListOffsetArray64(True)
                return ak.contents.RegularArray(
                    node.content[: node.length * size],
                    size,
                    zeros_length=node.length,
                    parameters=node.parameters,
                )
            return None

        layout = ak._do.recursively_apply(layout, action)
    return layout


def _select(layout, axis, k, ascending, *, indices):
    operation = ak._do.argsort if indices else ak._do.sort
    if axis == 0:
        return operation(layout, 0, ascending, True)[:k]

    def action(node, depth, **kwargs):
        if node.is_list and depth == axis:
            return operation(node, 1, ascending, True)[:, :k]
        return None

    return ak._do.recursively_apply(layout, action, numpy_to_regular=True)


def _align(layout, axis):
    counts = {}
    widths = {}
    original = layout
    for depth in range(axis + 1, layout.minmax_depth[1]):
        sizes = ak.num(original, axis=depth, highlevel=False)
        counts[depth] = ak.max(sizes, axis=axis, mask_identity=False, highlevel=False)
        if layout.backend.nplike.known_data:
            width = ak.max(sizes, axis=None, mask_identity=True)
            widths[depth] = 0 if width is None else int(width)
        else:
            widths[depth] = unknown_length
        # Missing lists have no coordinates. Turn them into empty lists before
        # padding, so gaps become missing numeric slots rather than missing rows.
        layout = ak.fill_none(
            layout,
            ak.contents.EmptyArray(backend=layout.backend),
            axis=depth - 1,
            highlevel=False,
        )
        layout = ak.pad_none(
            layout, widths[depth], axis=depth, clip=True, highlevel=False
        )
    return layout, counts, widths


def _trim_alignment(layout, counts, widths, axis):
    for depth in reversed(counts):
        sizes = counts[depth]
        # The count template has the searched axis reduced away. Pad the
        # remaining inner dimensions, then reinsert a singleton search axis.
        for inner in range(axis + 1, depth):
            sizes = ak.fill_none(
                sizes,
                ak.contents.EmptyArray(backend=sizes.backend),
                axis=inner - 2,
                highlevel=False,
            )
            sizes = ak.pad_none(
                sizes, widths[inner], axis=inner - 1, clip=True, highlevel=False
            )
        sizes = ak.fill_none(sizes, 0, axis=-1, highlevel=False)
        insertion = (slice(None),) * axis + (None,)
        sizes = sizes[insertion]
        local = ak.local_index(layout, axis=depth, highlevel=False)
        sizes, local = ak.broadcast_arrays(
            sizes, local, right_broadcast=False, highlevel=False
        )
        layout = layout[ak.Array(local) < ak.Array(sizes)]
    return layout
