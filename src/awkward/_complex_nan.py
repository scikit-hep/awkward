# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._backends.numpy import NumpyBackend
from awkward._layout import HighLevelContext
from awkward._namedaxis import _get_named_axis
from awkward._nplikes.numpy import Numpy

# CPU ufunc reductions with where are not exposed by NumpyLike.
np = Numpy.instance()._module


def try_reduce_dense(
    array, axis, keepdims, mask_identity, highlevel, behavior, attrs, *, mean
):
    """Reduce ordinary CPU matrices without copying valid values."""
    if type(axis) is not int or axis not in (-1, 1):
        return None

    candidate = array.layout if isinstance(array, ak.Array) else array
    if type(candidate) not in (
        np.ndarray,
        ak.contents.NumpyArray,
        ak.contents.RegularArray,
        ak.contents.ListOffsetArray,
    ):
        return None

    if type(candidate) in (np.ndarray, ak.contents.NumpyArray):
        dtype = candidate.dtype
    elif type(candidate.content) is ak.contents.NumpyArray:
        dtype = candidate.content.dtype
    else:
        return None
    if dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        return None

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False, primitive_policy="error")

    if (
        layout.backend is not NumpyBackend.instance()
        or layout._parameters
        or _get_named_axis(ctx)
    ):
        return None
    if type(layout) is ak.contents.NumpyArray:
        data = layout.data
    else:
        content = layout.content
        if (
            type(content) is not ak.contents.NumpyArray
            or content._parameters
            or type(content.data) is not np.ndarray
            or content.data.ndim != 1
            or not content.data.flags.c_contiguous
        ):
            return None
        if layout.is_regular:
            width = layout.size
            data = content.data[: layout.length * width].reshape(layout.length, width)
        else:
            offsets = layout.offsets.data
            if type(offsets) is not np.ndarray or len(offsets) < 2:
                return None
            width = int(offsets[1] - offsets[0])
            if (
                width < 0
                or offsets[0] < 0
                or offsets[-1] > content.length
                or not np.all(np.diff(offsets) == width)
            ):
                return None
            data = content.data[offsets[0] : offsets[-1]].reshape(layout.length, width)
    if (
        type(data) is not np.ndarray
        or data.ndim != 2
        or not data.size
        or not data.flags.c_contiguous
    ):
        return None

    # Addition propagates either NaN component. A NaN-free row sum proves
    # that all inputs in that row were valid, avoiding a full input-mask scan.
    # Infinities can also produce NaN sums; those rows take the masked path.
    if data.shape[1] < 128 and data.size >= 4096:
        sums = ak.operations.ak_sum._impl(
            layout, axis, False, False, False, ctx.behavior, ctx.attrs
        ).data
    else:
        with np.errstate(invalid="ignore", over="ignore"):
            sums = np.add.reduce(data, axis=-1)
    all_valid = not np.isnan(sums).any()
    if not all_valid:
        valid = np.isnan(data)
        np.logical_not(valid, out=valid)
        with np.errstate(invalid="ignore", over="ignore"):
            sums = np.add.reduce(data, axis=-1, where=valid)

    if mean:
        counts = (
            np.full(len(sums), data.shape[1], dtype=np.int64)
            if all_valid
            else np.count_nonzero(valid, axis=-1)
        )
        empty = counts == 0
        values = np.full(len(sums), np.nan, dtype=np.complex128)
        with np.errstate(invalid="ignore", divide="ignore"):
            np.divide(sums, counts, out=values, where=~empty)
        if mask_identity:
            index = np.arange(len(sums), dtype=np.int64)
            index[empty] = -1
            out = ak.contents.IndexedOptionArray(
                ak.index.Index64(index), ak.contents.NumpyArray(values)
            )
        else:
            out = ak.contents.NumpyArray(values)
    else:
        out = ak.contents.NumpyArray(sums)
        if mask_identity:
            out = ak.contents.ByteMaskedArray(
                ak.index.Index8(
                    np.zeros(len(sums), dtype=np.int8)
                    if all_valid
                    else ~valid.any(axis=-1)
                ),
                out,
                valid_when=False,
            )

    if keepdims:
        out = ak.contents.RegularArray(out, 1)
    return ctx.wrap(out, highlevel=highlevel)
