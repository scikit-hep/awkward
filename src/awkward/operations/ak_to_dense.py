# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from functools import partial

import awkward as ak
from awkward._backends.numpy import NumpyBackend
from awkward._dispatch import high_level_function
from awkward._experimental import experimental

__all__ = ("to_dense",)


@high_level_function()
@experimental
def to_dense(array, *, fill_value=0, clip=False, dtype=None):
    """Convert nested numeric data into a contiguous, rectangular NumPy array.

    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        fill_value: A numeric scalar used for missing values and padded entries.
        clip (bool): If False, pad each list axis to its longest list. If True,
            truncate each list axis to its shortest list. Empty lists participate
            in this minimum; missing lists do not.
        dtype: Optional NumPy numeric dtype for the result. By default, use a
            common dtype for the data and `fill_value`.

    Unlike #ak.to_numpy, this operation can change data: lists are padded or
    truncated and missing values are replaced. The outer array length is
    preserved. Each inner list axis uses one size across the whole array,
    including all record fields. Missing lists are filled at the chosen sizes.

    Record and tuple fields become a final feature axis, in field order. Nested
    records are flattened in depth-first field order. Fields must have the same
    number of list dimensions; an entirely missing field can adopt the dimensions
    of the other fields. A numeric value replacing an entire record is repeated
    across its features. Unions of records use fields in first-appearance order;
    absent fields are filled. Strings and other nonnumeric data are not supported.

    For example,

        >>> array = ak.Array([[{"x": 1, "y": 1.5}], [], [{"x": 2, "y": 2.5}, {"x": 3, "y": 3.5}]])
        >>> ak.to_dense(array).tolist()
        [[[1.0, 1.5], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [[2.0, 2.5], [3.0, 3.5]]]

    The result is an ordinary C-contiguous NumPy ndarray, with no mask, object
    dtype, or structured dtype. GPU input is copied to the CPU. Typetracer input
    is not supported, since choosing list sizes requires concrete data.
    """
    yield (array,)
    return _impl(array, fill_value, clip, dtype)


def _impl(array, fill_value, clip, dtype):
    import numpy  # noqa: TID251

    fill = numpy.asarray(fill_value)
    if fill.ndim != 0 or fill.dtype.kind not in "biufc":
        raise TypeError("fill_value must be a numeric scalar")
    if dtype is not None:
        dtype = numpy.dtype(dtype)
        if dtype.kind not in "biufc":
            raise TypeError("dtype must be a numeric NumPy dtype")

    layout = ak.to_layout(array, allow_record=False)
    if not layout.backend.nplike.known_data:
        raise TypeError("ak.to_dense requires concrete data")
    layout = layout.to_backend(NumpyBackend.instance())

    def columns(node):
        fields = []
        found_record = False

        def inspect(content, **kwargs):
            nonlocal found_record
            if content.parameter("__array__") in ("string", "bytestring"):
                raise TypeError("ak.to_dense only supports numeric data")
            if content.is_numpy and content.dtype.kind not in "biufc":
                raise TypeError("ak.to_dense only supports numeric data")
            if content.is_record:
                found_record = True
                for field in content.fields:
                    if field not in fields:
                        fields.append(field)
                return content
            return None

        ak._do.recursively_apply(node, inspect, return_array=False)
        if not found_record:
            return [node], False
        if not fields:
            raise TypeError("ak.to_dense cannot infer features from empty records")

        result = []
        for field in fields:

            def project(content, *, field=field, **kwargs):
                if content.is_record:
                    if field in content.fields:
                        return content.content(field)
                    return ak.contents.IndexedOptionArray(
                        ak.index.Index64(
                            numpy.full(content.length, -1, dtype=numpy.int64)
                        ),
                        ak.contents.EmptyArray(),
                    )
                return None

            projected = ak.transform(project, node, highlevel=False)
            result.extend(columns(projected)[0])
        return result, True

    layouts, has_records = columns(layout)

    def has_values(node):
        found = False

        def inspect(content, **kwargs):
            nonlocal found
            if content.is_numpy:
                found = True

        ak._do.recursively_apply(node, inspect, return_array=False)
        return found

    ranks = []
    unknown = []
    for column in layouts:
        low, high = column.minmax_depth
        if low != high:
            raise ValueError(
                "ak.to_dense requires a consistent number of list dimensions"
            )
        ranks.append(high)
        unknown.append(not has_values(column))
    rank = max(ranks)
    if any(r != rank and not empty for r, empty in zip(ranks, unknown, strict=True)):
        raise ValueError("ak.to_dense record fields must have the same list dimensions")

    def collect_regular_sizes(content, depth, *, axis, sizes, **kwargs):
        if depth == axis:
            if content.is_regular:
                sizes.append(content.size)
            elif content.is_numpy and len(content.shape) > 1:
                sizes.append(content.shape[1])

    sizes = []
    for axis in range(1, rank):
        observed = []
        regular = []
        for column, column_rank in zip(layouts, ranks, strict=True):
            if axis >= column_rank:
                continue
            counts = ak.num(column, axis=axis, highlevel=False)
            size = (ak.min if clip else ak.max)(counts, axis=None, mask_identity=True)
            if size is not None:
                observed.append(int(size))

            ak._do.recursively_apply(
                column,
                partial(collect_regular_sizes, axis=axis, sizes=regular),
                numpy_to_regular=True,
                return_array=False,
            )
        candidates = observed or regular
        sizes.append((min if clip else max)(candidates) if candidates else 0)

    arrays = []
    for column, column_rank in zip(layouts, ranks, strict=True):
        for axis, size in enumerate(sizes[: column_rank - 1], 1):
            column = ak.pad_none(column, size, axis=axis, clip=True, highlevel=False)
        arrays.append(ak.to_numpy(column, allow_missing=True))

    if dtype is None:
        dtype = numpy.result_type(
            *(x.dtype for x in arrays), numpy.min_scalar_type(fill_value)
        )
    filled = []
    shape = (layout.length, *sizes)
    for numpy_array, column_rank in zip(arrays, ranks, strict=True):
        values = numpy.ma.asarray(numpy_array, dtype=dtype).filled(fill_value)
        if column_rank < rank:
            values = numpy.broadcast_to(
                values.reshape((*values.shape, *((1,) * (rank - column_rank)))), shape
            )
        filled.append(values)
    result = numpy.stack(filled, axis=-1) if has_records else filled[0]
    return numpy.ascontiguousarray(result)
