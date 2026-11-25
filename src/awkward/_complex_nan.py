# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._backends.numpy import NumpyBackend
from awkward._layout import HighLevelContext
from awkward._namedaxis import _get_named_axis
from awkward._nplikes.numpy import Numpy

# This fast path only operates on ordinary CPU NumPy buffers.
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

    sums = np.empty(data.shape[0], dtype=data.dtype)
    counts = np.empty(data.shape[0], dtype=np.int64)
    offsets = np.arange(data.shape[0] + 1, dtype=np.int64) * data.shape[1]
    component = np.float32 if data.dtype == np.dtype(np.complex64) else np.float64
    layout.backend.maybe_kernel_error(
        layout.backend[
            "awkward_reduce_nansum_count_complex",
            component,
            np.int64,
            component,
            np.int64,
        ](
            sums.view(component),
            counts,
            data.reshape(-1).view(component),
            offsets,
            data.shape[0],
        )
    )

    if mean:
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
                ak.index.Index8(counts == 0),
                out,
                valid_when=False,
            )

    if keepdims:
        out = ak.contents.RegularArray(out, 1)
    return ctx.wrap(out, highlevel=highlevel)
